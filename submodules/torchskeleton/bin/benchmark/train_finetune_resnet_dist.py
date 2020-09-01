# -*- coding: utf-8 -*-
import os
import sys
import logging
import copy

import torch
import torchvision
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)
import skeleton
import datasets
from theconf import Config, ConfigArgumentParser, AverageMeter

LOGGER = logging.getLogger(__name__)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--datapath', type=str, default='./data')

    parser.add_argument('--init-model', type=str, required=True)
    parser.add_argument('-c', '--checkpoint', type=str, required=True)

    parser.add_argument('-a', '--architecture', type=str, default='resnet-50',
                        help='resnet-[18|34|50|101], default:resnet-50')
    parser.add_argument('--sync-bn', action='store_true')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--batch', type=int, default=256)

    parser.add_argument('--loss_label_smooth', type=float, default=0.0)
    parser.add_argument('--schedule', type=str, default='multistep')
    parser.add_argument('--epoch', type=int, default=120)

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

    assert 'resnet' in args.architecture
    assert args.architecture.split('-')[1] in ['18', '34', '50', '101']

    if args.local_rank >= 0:
        logging.info('Distributed: wait dist process group:%d', args.local_rank)
        dist.init_process_group(backend=args.dist_backend, init_method='env://',
                                world_size=int(os.environ['WORLD_SIZE']))
        assert (int(os.environ['WORLD_SIZE']) == dist.get_world_size())
        logging.info('Distributed: success device:%d (%d/%d)', args.local_rank, dist.get_rank(), dist.get_world_size())

        world_size = dist.get_world_size()
        world_rank = dist.get_rank()
        num_gpus = 1
    else:
        logging.info('Single process')
        args.local_rank = 0
        world_size = 1
        world_rank = 0
        num_gpus = torch.cuda.device_count()

    if world_rank == 0:
        summary_writers = {
            'train': SummaryWriter('%s/train' % args.checkpoint),
            'valid': SummaryWriter('%s/valid' % args.checkpoint),
            'valid_ema': SummaryWriter('%s/valid_ema' % args.checkpoint),
        }

    environments = skeleton.utils.Environments()
    device = torch.device('cuda', args.local_rank)
    torch.cuda.set_device(device)
    LOGGER.info('environemtns\n%s', environments)
    LOGGER.info('args\n%s', args)

    batch = args.batch * num_gpus * (2 if args.half else 1)
    total_batch = batch * world_size
    steps_per_epoch = int(1281167 // total_batch)
    LOGGER.info('other stats\nnum_gpus:%d\nbatch:%d\ntotal_batch:%d\nsteps_per_epoch:%d',
                num_gpus, batch, total_batch, steps_per_epoch)

    input_size = 224
    num_classes = 1000
    norm_layer = torch.nn.SyncBatchNorm if world_size > 1 and args.sync_bn else torch.nn.BatchNorm2d
    model = torchvision.models.resnet18(norm_layer=norm_layer, zero_init_residual=True) if '18' in args.architecture else None
    model = torchvision.models.resnet34(norm_layer=norm_layer, zero_init_residual=True) if '34' in args.architecture else model
    model = torchvision.models.resnet50(norm_layer=norm_layer, zero_init_residual=True) if '50' in args.architecture else model
    model = torchvision.models.resnet101(norm_layer=norm_layer, zero_init_residual=True) if '101' in args.architecture else model
    model = model.to(device=device)

    checkpoint = torch.load(args.init_model, map_location=lambda storage, location: storage)
    checkpoint['model'].pop('fc.weight')
    checkpoint['model'].pop('fc.bias')
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
    LOGGER.info('load model:%s (missing:%s, unexpected:%s)', args.checkpoint, missing_keys, unexpected_keys)

    if args.half:
        for module in model.modules():
            if not isinstance(module, (torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm)):
                module.half()
    model_ema_eval = skeleton.nn.ExponentialMovingAverage(
        copy.deepcopy(model), mu=0.9999, data_parallel=world_size == 1
    ).to(device=device).eval()

    # batchnorm parameter annealing
    def update_bn_params(momentum=0.9):
        def update(module):
            if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm)):
                module.momentum = momentum
        return update

    epochs = args.epoch
    learning_rate = 0.01
    weight_decay = 1e-4
    # original optimizer
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    # lr per layer optimizer
    optimizer = torch.optim.SGD(
        [
            {'lr': 0.0005, 'params': [param for name, param in model.named_parameters() if name in ['conv1.weight', 'bn1.weight', 'bn1.bias']]},
            {'lr': 0.0001, 'params': [param for name, param in model.named_parameters() if name.startswith('layer1')]},
            {'lr': 0.0005, 'params': [param for name, param in model.named_parameters() if name.startswith('layer2')]},
            {'lr': 0.001, 'params': [param for name, param in model.named_parameters() if name.startswith('layer3')]},
            {'lr': 0.005, 'params': [param for name, param in model.named_parameters() if name.startswith('layer4')]},
            {'lr': 0.01, 'params': [param for name, param in model.named_parameters() if name.startswith('fc')]},
        ],
        lr=learning_rate, momentum=0.9, weight_decay=weight_decay, nesterov=True
    )
    if args.schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [(l + 5) * steps_per_epoch for l in [30, 60, 90, 110]], gamma=0.1)
    elif args.schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch, eta_min=1e-6)
    else:
        raise ValueError('not support schedule: %s', args.schedule)

    scheduler = skeleton.optim.GradualWarmup(optimizer, scheduler, steps=5 * steps_per_epoch, multiplier=total_batch / 256)

    loss_scaler = 1.0 if not args.half else 1024.0
    if args.loss_label_smooth > 0.0:
        loss_fn = skeleton.nn.CrossEntropyLabelSmooth(num_classes=1000, epsilon=0.1, reduction='mean')
    else:
        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    def criterion(logits, targets):
        return loss_scaler * loss_fn(logits, targets)

    metricer = skeleton.nn.AccuracyMany((1, 5))
    meter = AverageMeter('loss', 'accuracy', 'accuracy_top5').to(device=device)
    def metrics_calculator(logits, targets):
        with torch.no_grad():
            loss = criterion(logits, targets)
            top1, top5 = metricer(logits, targets)
            return {
                'loss': loss.detach() / loss_scaler,
                'accuracy': top1.detach(),
                'accuracy_top5': top5.detach()
            }

    # profiler = skeleton.nn.Profiler(model)
    # params = profiler.params()
    # flops = profiler.flops(torch.ones(1, 3, input_size, input_size, dtype=torch.float, device=device))
    # LOGGER.info('arechitecture\n%s\ninput:%d\nprarms:%.2fM\nGFLOPs:%.3f', args.architecture, input_size, params / (1024 * 1024), flops / (1024 * 1024 * 1024))

    LOGGER.info('arechitecture:%s\ninput:%d\nnum_classes:%d', args.architecture, input_size, num_classes)
    LOGGER.info('optimizers\nloss:%s\noptimizer:%s\nscheduler:%s\nloss_scaler:%f',
                str(criterion), str(optimizer), str(scheduler), loss_scaler)
    LOGGER.info('hyperparams\nbatch:%d\ninput_size:%d\nsteps_per_epoch:%d\nlearning_rate_init:%.4f',
                batch, input_size, steps_per_epoch, learning_rate)

    dataset = datasets.ImageNet(
        split='train',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(input_size, scale=(0.05, 1.0), interpolation=Image.BICUBIC),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ColorJitter(brightness=0.12,
                                               contrast=0.5,
                                               saturation=0.5,
                                               hue=0.2),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
            # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    )
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=world_rank, shuffle=True)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch, sampler=sampler,
        num_workers=args.workers, drop_last=True, pin_memory=True
    )
    steps = len(dataloader)

    resize_image = input_size + 32
    dataset = datasets.ImageNet(
        split='val',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(resize_image, interpolation=Image.BICUBIC),
            torchvision.transforms.CenterCrop(input_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
            # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    )
    dataloader_val = torch.utils.data.DataLoader(
        dataset, batch_size=batch, shuffle=False,
        num_workers=args.workers, drop_last=False, pin_memory=True
    )

    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
        for param in model.parameters():
            dist.broadcast(param.data, 0)
    else:
        model = torch.nn.parallel.DataParallel(model)
    torch.cuda.synchronize()

    best_accuracy = 0.0
    timer('init', reset_step=True, exclude_total=True)
    for epoch in range(epochs):
        model.train()
        sampler.set_epoch(epoch)  # re-shuffled dataset per node

        model.apply(update_bn_params(max(1.0-(10.0/(epoch+1)), 0.9)))
        for step, (inputs, targets) in zip(range(steps), dataloader):
            timer('init', reset_step=True, exclude_total=True)
            inputs = inputs.to(device=device, dtype=torch.half if args.half else torch.float, non_blocking=True)
            targets = targets.to(device=device, non_blocking=True)

            logits = model(inputs).to(dtype=torch.float)
            loss = criterion(logits, targets)
            timer('forward')

            optimizer.zero_grad()
            loss.backward()
            if loss_scaler != 1.0:
                for param in model.parameters():
                    param.grad.data /= loss_scaler
            timer('backward')

            optimizer.step()
            scheduler.step()

            model_ema_eval.update(model.module, step=epoch * steps + step)
            timer('optimize')

            meter.updates(metrics_calculator(logits, targets))

            if step % (steps // 100) == 0:
                LOGGER.info(
                    '[train] [rank:%03d] %03d/%03d epoch (%02d%%) | lr:%s | %s',
                    world_rank, epoch, epochs, 100.0 * step / steps,
                    ','.join(['%.5f'%lr for lr in scheduler.get_lr()]),
                    str(meter)
                )
            timer('remain')

        metrics_train = meter.get()
        logging.info(
            '[train] [rank:%03d] %03d/%03d epoch | %s',
            world_rank, epoch, epochs, str(meter)
        )
        if world_rank == 0:
            print('[train] [rank:%03d] %03d/%03d epoch | %s' % (
                world_rank, epoch, epochs, str(meter)
            ))
        meter.reset(step=epoch)

        is_best = False
        metrics_valid = {}
        metrics_valid_ema = {}
        if not world_rank in [0]:
            LOGGER.info('[valid] [rank:%03d] wait master', world_rank)
        elif epoch % args.valid_skip == 0 or epoch > (epochs * 0.9):
            for name, m in [('valid', model), ('valid_ema', model_ema_eval)]:
                m.eval()
                with torch.no_grad():
                    for inputs, targets in dataloader_val:
                        num_samples = inputs.shape[0]
                        inputs = inputs.to(device=device, dtype=torch.half if args.half else torch.float, non_blocking=True)
                        targets = targets.to(device=device, non_blocking=True)

                        logits = m(inputs)
                        meter.updates(metrics_calculator(logits, targets), n=num_samples)

                if name == 'valid':
                    metrics_valid = meter.get()
                else:
                    metrics_valid_ema = meter.get()

                print('[%s] [rank:%03d] %03d/%03d epoch | %s' % (name, world_rank, epoch, epochs, str(meter)))
                is_best = best_accuracy < metrics_valid['accuracy']
                best_accuracy = max(best_accuracy, metrics_valid['accuracy'])

                meter.reset(step=epoch)
        else:
            LOGGER.info('[valid] [rank:%03d] skip master', world_rank)

        if world_rank == 0:
            throughput = (epoch + 1) * steps * batch * world_size * timer.throughput()
            summary_writers['train'].add_scalar('hyperparams/lr', scheduler.get_lr()[-1], global_step=epoch)
            summary_writers['train'].add_scalar('performance/throughput', throughput, global_step=epoch)
            for key, value in metrics_train.items():
                summary_writers['train'].add_scalar('metrics/%s' % key, value, global_step=epoch)
            for key, value in metrics_valid.items():
                summary_writers['valid'].add_scalar('metrics/%s' % key, value, global_step=epoch)
            for key, value in metrics_valid_ema.items():
                summary_writers['valid_ema'].add_scalar('metrics/%s' % key, value, global_step=epoch)

            LOGGER.info('[train] [rank:%03d] %03d/%03d epoch | throughput:%.4f images/sec, %.4f sec/epoch',
                        world_rank, epoch, epochs, throughput, timer.total_time / (epoch + 1))
            skeleton.utils.save_checkpoints(
                epoch, '%s/checkpoints' % args.checkpoint,
                {
                    'epoch': epoch,
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'metrics': {
                        'train': metrics_train,
                        'valid': metrics_valid
                    }
                },
                is_best=is_best, keep_last=30
            )


if __name__ == '__main__':
    # single node / multi gpu
    # > python bin/benchmark/train_resnet_dist.py -c runs/test
    # multi node
    # > python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --master_addr=task1 --master_port=2901 --node_rank=0 bin/benchmark/train_resnet_dist.py -c ./runs/test
    # > python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --master_addr=task1 --master_port=2901 --node_rank=1 bin/benchmark/train_resnet_dist.py -c ./runs/test
    # ...
    # > python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --master_addr=task1 --master_port=2901 --node_rank=3 bin/benchmark/train_resnet_dist.py -c ./runs/test
    main()
