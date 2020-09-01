# -*- coding: utf-8 -*-
import os
import sys
import logging

import torch
import torch.distributed as dist


base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)
import skeleton
import efficientnet


LOGGER = logging.getLogger(__name__)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-a', '--architecture', type=str, default='efficientnet-b4')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--batch', type=int, default=None)
    parser.add_argument('--steps', type=int, default=50)

    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='Used for multi-process training. Can either be manually set ' +
                             'or automatically set by using \'python -m multiproc\'.')

    parser.add_argument('--log-filename', type=str, default='')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def main():
    timer = skeleton.utils.Timer()

    args = parse_args()
    log_format = '[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)03d] %(message)s'
    level = logging.DEBUG if args.debug else logging.INFO
    if not args.log_filename:
        logging.basicConfig(level=level, format=log_format, stream=sys.stderr)
    else:
        logging.basicConfig(level=level, format=log_format, filename=args.log_filename)

    torch.backends.cudnn.benchmark = True

    assert 'efficientnet' in args.architecture
    assert args.architecture.split('-')[1] in ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7']

    logging.info('Distributed: wait dist process group:%d', args.local_rank)
    dist.init_process_group(backend=args.dist_backend, init_method='env://', world_size=int(os.environ['WORLD_SIZE']))
    assert (int(os.environ['WORLD_SIZE']) == dist.get_world_size())
    world_size = dist.get_world_size()
    logging.info('Distributed: success device:%d (%d/%d)', args.local_rank, dist.get_rank(), dist.get_world_size())

    environments = skeleton.utils.Environments()
    device = torch.device('cuda', args.local_rank)
    torch.cuda.set_device(device)

    if args.batch is None:
        args.batch = 128 if 'b0' in args.architecture else args.batch
        args.batch = 96 if 'b1' in args.architecture else args.batch
        args.batch = 64 if 'b2' in args.architecture else args.batch
        args.batch = 32 if 'b3' in args.architecture else args.batch
        args.batch = 16 if 'b4' in args.architecture else args.batch
        args.batch = 8 if 'b5' in args.architecture else args.batch
        args.batch = 6 if 'b6' in args.architecture else args.batch
        args.batch = 4 if 'b7' in args.architecture else args.batch
        args.batch *= 2

    input_size = efficientnet.EfficientNet.get_image_size(args.architecture)
    model = efficientnet.EfficientNet.from_name(args.architecture).to(device=device)

    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5, nesterov=True)

    profiler = skeleton.nn.Profiler(model)
    params = profiler.params()
    flops = profiler.flops(torch.ones(1, 3, input_size, input_size, dtype=torch.float, device=device))

    LOGGER.info('environemtns\n%s', environments)
    LOGGER.info('arechitecture\n%s\ninput:%d\nprarms:%.2fM\nGFLOPs:%.3f', args.architecture, input_size, params / (1024 * 1024), flops / (1024 * 1024 * 1024))
    LOGGER.info('optimizers\nloss:%s\noptimizer:%s', str(criterion), str(optimizer))
    LOGGER.info('args\n%s', args)

    # model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpus)))
    batch = args.batch * (2 if args.half else 1)
    inputs = torch.ones(batch, 3, input_size, input_size, dtype=torch.float, device=device)
    targets = torch.zeros(batch, dtype=torch.long, device=device)

    if args.half:
        inputs = inputs.half()
        for module in model.modules():
            if not isinstance(module, torch.nn.BatchNorm2d):
                module.half()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # warmup
    for _ in range(2):
        logits = model(inputs)
        loss = criterion(logits, targets)

        loss.backward()
        model.zero_grad()
    torch.cuda.synchronize()

    timer('init', reset_step=True, exclude_total=True)
    for step in range(args.steps):
        timer('init', reset_step=True)

        logits = model(inputs)
        loss = criterion(logits, targets)
        timer('forward')

        loss.backward()
        timer('backward')

        optimizer.step()
        optimizer.zero_grad()
        timer('optimize')

        LOGGER.info('[%02d] %s', step, timer)

    if dist.get_rank() == 0:
        images = args.steps * batch * world_size
        LOGGER.info('throughput:%.4f images/sec', images * timer.throughput())


if __name__ == '__main__':
    # single node
    # > python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 bin/benchmark/efficientnet_throughput_dist.py
    # multi node
    # > python -m torch.distributed.launch --nproc_per_node=4 --nnodes=4 --master_addr=MASTER_HOST_NAME --master_port=MASTER_PORT --node_rank=0 bin/benchmark/efficientnet_throughput_dist.py
    main()
