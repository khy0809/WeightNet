# -*- coding: utf-8 -*-
import os
import sys
import copy
import math
import logging

import torch
import torchvision
import torch.distributed as dist
from PIL import Image


base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)
import skeleton
import datasets
import efficientnet


LOGGER = logging.getLogger(__name__)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--datapath', type=str, default='./data')

    parser.add_argument('-c', '--checkpoint', type=str, default=None)

    parser.add_argument('-a', '--architecture', type=str, default='efficientnet-b4', help='efficientnet-[b0|b1|..|b7] or resnet=[18|34|50|101], default:efficientnet-b7')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--batch', type=int, default=None)
    parser.add_argument('--epoch', type=int, default=350)
    parser.add_argument('--sync-bn', action='store_true')

    parser.add_argument('--valid-skip', type=int, default=1)
    parser.add_argument('--seed', type=lambda x: int(x, 0), default=None)

    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')

    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='Used for multi-process training. Can either be manually set ' +
                             'or automatically set by using \'python -m multiproc\'.')

    parser.add_argument('--log-filename', type=str, default='')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


class GradualWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, scheduler, steps, last_epoch=-1):
        self.scheduler = scheduler
        self.steps = steps
        super(GradualWarmup, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * min(1.0, (self.last_epoch / self.steps))
                for base_lr in self.scheduler.get_lr()]

    def step(self, epoch=None):
        if self._step_count > 0:
            self.scheduler.step(epoch)
        super(GradualWarmup, self).step(epoch)

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ('
        format_string += '\n\toptimizer=' + self.optimizer.__class__.__name__ + ','
        format_string += '\n\tscheduler=' + self.scheduler.__class__.__name__ + ','
        format_string += '\n\tsteps=' + str(self.steps) + ','
        format_string += '\n\tlast_epoch=' + str(self.last_epoch)
        format_string += '\n)'
        return format_string


def main():
    timer = skeleton.utils.Timer()
    args = parse_args()
    if args.checkpoint is None:
        raise ValueError('must be a set --checkpoint')

    log_format = '[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)03d] %(message)s'
    level = logging.DEBUG if args.debug else logging.INFO
    if not args.log_filename:
        logging.basicConfig(level=level, format=log_format, stream=sys.stderr)
    else:
        logging.basicConfig(level=level, format=log_format, filename=args.log_filename)
    torch.backends.cudnn.benchmark = True
    if args.seed is not None:
        skeleton.utils.set_random_seed_all(args.seed, deterministic=False)

    assert 'efficientnet' in args.architecture or 'resnet' in args.architecture
    assert args.architecture.split('-')[1] in ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7'] or\
           args.architecture.split('-')[1] in ['18', '34', '50', '101']

    if args.local_rank >= 0:
        logging.info('Distributed: wait dist process group:%d', args.local_rank)
        dist.init_process_group(backend=args.dist_backend, init_method='env://', world_size=int(os.environ['WORLD_SIZE']))
        assert (int(os.environ['WORLD_SIZE']) == dist.get_world_size())
        logging.info('Distributed: success device:%d (%d/%d)', args.local_rank, dist.get_rank(), dist.get_world_size())

        world_size = dist.get_world_size()
        world_rank = dist.get_rank()
    else:
        logging.info('Single proces')
        args.local_rank = 0
        world_size = 1
        world_rank = 0

    environments = skeleton.utils.Environments()
    device = torch.device('cuda', args.local_rank)
    torch.cuda.set_device(device)

    if args.batch is None:
        if 'efficientnet' in args.architecture:
            batch = 128 if 'b0' in args.architecture else 1
            batch = 96 if 'b1' in args.architecture  else batch
            batch = 64 if 'b2' in args.architecture  else batch
            batch = 32 if 'b3' in args.architecture  else batch
            batch = 16 if 'b4' in args.architecture  else batch
            batch = 8 if 'b5' in args.architecture   else batch
            batch = 6 if 'b6' in args.architecture   else batch
            batch = 4 if 'b7' in args.architecture   else batch
            batch *= 2
        else:
            batch = 256
        batch = batch * (2 if args.half else 1)
    else:
        batch = args.batch

    LOGGER.info('environemtns\n%s', environments)
    LOGGER.info('args\n%s', args)

    total_batch = batch * world_size
    steps_per_epoch = int(1281167 / total_batch)
    
    if 'efficientnet' in args.architecture:
        norm_layer = torch.nn.SyncBatchNorm if world_size > 1 and args.sync_bn else torch.nn.BatchNorm2d
        input_size = efficientnet.EfficientNet.get_image_size(args.architecture)
        model = efficientnet.EfficientNet.from_name(args.architecture,
                                                    # override_params={'batch_norm_momentum': 0.9999},
                                                    norm_layer=norm_layer).to(device=device)
        # model.set_swish(memory_efficient=False)

        def kernel_initializer(module):
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(module.weight,  mode='fan_in', nonlinearity='linear')
        model.apply(kernel_initializer)

        epoch_scale = 1  # if args.architecture.split('-')[1] in ['b5', 'b6', 'b7'] else 2
        epochs = 350 * epoch_scale
        learning_rate = 0.256 / (4096 / total_batch)
        weight_decay = 1e-5
        # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate,
        #                                 alpha=0.9, momentum=0.9, weight_decay=0.0,
        #                                 eps=math.sqrt(0.001))
        optimizer = skeleton.optim.RMSprop(model.parameters(), lr=learning_rate,
                                        alpha=0.9, momentum=0.9, weight_decay=0.0,
                                        eps=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(2.4 * epoch_scale * steps_per_epoch), gamma=0.97)

        criterion = skeleton.nn.CrossEntropyLabelSmooth(num_classes=1000, epsilon=0.1, reduction='mean')
    else:
        input_size = 224
        norm_layer = torch.nn.SyncBatchNorm if world_size > 1 and args.sync_bn else torch.nn.BatchNorm2d
        model = torchvision.models.resnet18(norm_layer=norm_layer) if '18' in args.architecture else None
        model = torchvision.models.resnet34(norm_layer=norm_layer) if '34' in args.architecture else model
        model = torchvision.models.resnet50(norm_layer=norm_layer) if '50' in args.architecture else model
        model = torchvision.models.resnet101(norm_layer=norm_layer) if '101' in args.architecture else model
        model = model.to(device=device)

        epochs = 90
        learning_rate = 0.1 / (256 / total_batch)
        weight_decay = 1e-5
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0, nesterov=True)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [l * steps_per_epoch for l in [30, 60, 80]], gamma=0.1)

        criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    scheduler = GradualWarmup(optimizer, scheduler, steps=5 * steps_per_epoch)
    metricer = skeleton.nn.AccuracyMany((1, 5))

    # profiler = skeleton.nn.Profiler(model)
    # params = profiler.params()
    # flops = profiler.flops(torch.ones(1, 3, input_size, input_size, dtype=torch.float, device=device))

    # LOGGER.info('arechitecture\n%s\ninput:%d\nprarms:%.2fM\nGFLOPs:%.3f', args.architecture, input_size, params / (1024 * 1024), flops / (1024 * 1024 * 1024))
    LOGGER.info('arechitecture:%s\ninput:%d', args.architecture, input_size)
    LOGGER.info('optimizers\nloss:%s\noptimizer:%s\nscheduler:%s', str(criterion), str(optimizer), str(scheduler))
    LOGGER.info('hyperparams\nbatch:%d\ninput_size:%d\nsteps_per_epoch:%d\nlearning_rate_init:%.4f',
                batch, input_size, steps_per_epoch, learning_rate)

    # dataset = skeleton.data.ImageNet(root=args.datapath + '/imagenet', split='train', transform=torchvision.transforms.Compose([
    dataset = datasets.ImageNetCovered(
        split='train',
        special_transform=datasets.imagenet.RandomCropBBox(min_object_covered=0.1, scale=(0.08, 1.0), ratio=(3./4., 4./3.)),
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize((input_size, input_size), interpolation=Image.BICUBIC),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=world_rank)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch, sampler=sampler,
        num_workers=args.workers, drop_last=True, pin_memory=True
    )
    steps = len(dataloader)

    resize_image = input_size if 'efficientnet' in args.architecture else int(input_size * 1.14)
    # dataset = torchvision.datasets.ImageNet(root=args.datapath + '/imagenet', split='val', transform=torchvision.transforms.Compose([
    dataset = datasets.ImageNet(split='val', transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize(resize_image, interpolation=Image.BICUBIC),
        torchvision.transforms.CenterCrop(input_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    dataloader_val = torch.utils.data.DataLoader(
        dataset, batch_size=batch, shuffle=False,
        num_workers=args.workers, drop_last=False, pin_memory=True
    )

    if args.half:
        for module in model.modules():
            if not isinstance(module, torch.nn.BatchNorm2d):
                module.half()
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
        for param in model.parameters():
            dist.broadcast(param.data, 0)
    torch.cuda.synchronize()

    loss_scaler = 1.0 if not args.half else 1024.0

    params_without_bn = [params for name, params in model.named_parameters() if not ('_bn' in name or '.bn' in name)]

    best_accuracy = 0.0
    timer('init', reset_step=True, exclude_total=True)
    for epoch in range(epochs):
        model.train()
        sampler.set_epoch(epoch)  # re-shuffled dataset per node
        loss_sum = torch.zeros(1, device=device, dtype=torch.half if args.half else torch.float)
        accuracy_top1_sum = torch.zeros(1, device=device)
        accuracy_top5_sum = torch.zeros(1, device=device)
        for step, (inputs, targets) in enumerate(dataloader):
            timer('init', reset_step=True, exclude_total=True)
            inputs = inputs.to(device=device, dtype=torch.half if args.half else torch.float, non_blocking=True)
            targets = targets.to(device=device, non_blocking=True)

            logits = model(inputs).to(dtype=torch.float)
            loss = criterion(logits, targets)

            # l2 regularizer \wo batchnorm params
            loss_weight_l2 = sum([p.to(dtype=torch.float).norm(2) for p in params_without_bn])
            loss = loss + (weight_decay * loss_weight_l2)
            timer('forward')

            optimizer.zero_grad()
            if loss_scaler == 1.0:
                loss.backward()
            else:
                (loss * loss_scaler).backward()
                for param in model.parameters():
                    param.grad.data /= loss_scaler
            timer('backward')

            optimizer.step()
            scheduler.step()
            timer('optimize')

            with torch.no_grad():
                accuracies = metricer(logits, targets)

            loss_sum += loss.detach()
            accuracy_top1_sum += accuracies[0].detach()
            accuracy_top5_sum += accuracies[1].detach()

            if step % (steps // 100) == 0:
                LOGGER.info(
                    '[train] [rank:%03d] %03d/%03d epoch (%02d%%) | loss:%.4f, top1:%.4f, top5:%.4f | lr:%.4f',
                    world_rank, epoch, epochs, 100.0 * step / steps,
                    loss_sum.item() / (step + 1), accuracy_top1_sum.item() / (step + 1), accuracy_top5_sum.item() / (step + 1),
                    scheduler.get_lr()[0]
                )
            timer('remain')

        metrics = {
            'loss': loss_sum.item() / steps,
            'accuracy_top1': accuracy_top1_sum.item() / steps,
            'accuracy_top5': accuracy_top5_sum.item() / steps,
        }
        logging.info(
            '[train] [rank:%03d] %03d/%03d epoch | loss:%.5f, top1:%.4f, top5:%.4f',
            world_rank, epoch, epochs, metrics['loss'], metrics['accuracy_top1'], metrics['accuracy_top5']
        )

        is_best = False
        metrics_train = copy.deepcopy(metrics)
        if not world_rank == 0:
            LOGGER.info('[valid] [rank:%03d] wait master', world_rank)
        elif epoch % args.valid_skip == 0 or epoch > (epochs * 0.9):
            model.eval()

            num_samples_sum = 0
            loss_sum = torch.zeros(1, device=device, dtype=torch.half if args.half else torch.float)
            accuracy_top1_sum = torch.zeros(1, device=device)
            accuracy_top5_sum = torch.zeros(1, device=device)
            with torch.no_grad():
                for inputs, targets in dataloader_val:
                    num_sampels = inputs.shape[0]
                    inputs = inputs.to(device=device, dtype=torch.half if args.half else torch.float, non_blocking=True)
                    targets = targets.to(device=device, non_blocking=True)

                    logits = model(inputs)
                    loss = criterion(logits, targets)
                    accuracies = metricer(logits, targets)

                    num_samples_sum += num_sampels
                    loss_sum += loss.detach() * num_sampels
                    accuracy_top1_sum += accuracies[0].detach() * num_sampels
                    accuracy_top5_sum += accuracies[1].detach() * num_sampels

            metrics = {
                'loss': loss_sum.item() / num_samples_sum,
                'accuracy_top1': accuracy_top1_sum.item() / num_samples_sum,
                'accuracy_top5': accuracy_top5_sum.item() / num_samples_sum,
            }
            logging.info(
                '[valid] [rank:%03d] %02d/%02d epoch | loss:%.5f, top1:%.4f, top5:%.4f',
                world_rank, epoch, epochs, metrics['loss'], metrics['accuracy_top1'], metrics['accuracy_top5']
            )
            is_best = best_accuracy < metrics['accuracy_top1']
            best_accuracy = max(best_accuracy, metrics['accuracy_top1'])
        else:
            LOGGER.info('[valid] [rank:%03d] skip master', world_rank)

        if world_rank == 0:
            LOGGER.info('[train] [rank:%03d] %03d/%03d epoch | throughput:%.4f images/sec, %.4f sec/epoch',
                        world_rank, epoch, epochs,
                        (epoch + 1) * steps * batch * world_size * timer.throughput(),
                        timer.total_time / (epoch + 1))
            skeleton.utils.save_checkpoints(
                epoch, args.checkpoint,
                {
                    'epoch': epoch,
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'metrics': {
                        'train': metrics_train,
                        'valid': metrics
                    }
                },
                is_best=is_best, keep_last=30
            )


if __name__ == '__main__':
    # single node
    # > python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 bin/benchmark/train_efficientnet_dist.py -c ./runs/test
    # multi node
    # > python -m torch.distributed.launch --nproc_per_node=8 --nnodes=16 --master_addr=task1 --master_port=2901 --node_rank=0 bin/benchmark/train_efficientnet_dist.py -c ./runs/test
    # > python -m torch.distributed.launch --nproc_per_node=8 --nnodes=16 --master_addr=task1 --master_port=2901 --node_rank=0 bin/benchmark/train_efficientnet_dist.py -c ./runs/efficientnet-b4/batch4k/rmsprop/20191130/checkpoints --valid-skip 10 > log.node1.txt 2>&1 &
    main()
