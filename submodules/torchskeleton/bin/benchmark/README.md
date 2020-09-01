# Benchmark EfficientNet

## Result

### Accuracy

Architecture                | #Params | #GFLOPs | Val Top1 | Val Top5 | Train Top1 | Train Top5 | 2 GPUs Throughput (images/sec) |
----------------------------|--------:|--------:|---------:|---------:|-----------:|-----------:|------------------------:|
efficientnet b0 [paper]     |    5.3M |    0.39 |    76.3% |    93.2% |      N/A |      N/A |       N/A |
efficientnet b0 [reproduce] |    5.0M |    0.37 |    76.3% |    92.9% |    86.4% |    97.6% | 1388.4468 |
efficientnet b1 [paper]     |    7.8M |    0.70 |    78.8% |    94.4% |      N/A |      N/A |       N/A |
efficientnet b1 [reproduce] |    7.4M |    0.60 |    78.4% |    94.2% |    90.9% |    99.0% | 1026.0683 |
efficientnet b2 [paper]     |    9.2M |    1.00 |    79.8% |    94.9% |      N/A |      N/A |       N/A |
efficientnet b2 [reproduce] |    8.7M |    0.83 |    79.8% |    94.7% |    91.8% |    99.1% |  841.2940 |
efficientnet b3 [paper]     |     12M |    1.80 |    81.1% |    95.5% |      N/A |      N/A |       N/A |
efficientnet b3 [reproduce] |   11.7M |    1.56 |    81.2% |    95.4% |    94.6% |    99.6% |  514.2853 |
efficientnet b4 [paper]     |     19M |    4.20 |    82.6% |    96.3% |      N/A |      N/A |       N/A |
efficientnet b4 [reproduce] |   18.5M |    3.80 |    82.8% |    96.2% |    94.5% |    99.6% |  255.5461 |
efficientnet b5 [paper]     |     30M |    9.90 |    83.3% |    96.7% |      N/A |      N/A |       N/A |
efficientnet b5 [reproduce] |   29.0M |    9.01 |    83.8% |    96.8% |    95.2% |    99.7% |  126.2877 |
efficientnet b6 [paper]     |     43M |   19.00 |    84.0% |    96.9% |      N/A |      N/A |       N/A |
efficientnet b6 [reproduce] |   41.1M |   17.24 |    84.2% |    96.9% |    94.4% |    99.6% |   78.7448 |
efficientnet b7 [paper]     |     66M |   37.00 |    84.4% |    97.1% |      N/A |      N/A |       N/A |
efficientnet b7 [reproduce] |   63.3M |   33.58 |    84.5% |    97.0% |    96.1% |    99.8% |   44.7139 |


### Tranining Reproduce

Architecture               | augment  | Val Top1 | Val Top5 |
---------------------------|----------|----------:|--------:|
efficientnet b4 [original] | standard | 82.6% | - |
efficientnet b4 [original] | autoaug  | 83.0% | - |
efficientnet b4 [original] | randaug  | -  | - |
efficientnet b4 [original] | advprop  | 83.3% | - |



### Training Throughput (images/sec)

* Distributed Throughput (static data)

Architecture    | preci | metric      |   Batch |    8 GPUs |   16 GPUs |   32 GPUs |   64 GPUs |  128 GPUs |
----------------|-------|-------------|--------:|----------:|----------:|----------:|----------:|----------:|
efficientnet b4 |  full | images/sec  |  32 * N |  338.2595 |  671.5585 | 1266.5825 | 2494.8769 | 4313.9375 |
efficientnet b4 |  full | hours/epoch |  32 * N |    1.0520 |    0.5299 |    0.2810 |    0.1426 |    0.0824 |
efficientnet b4 |  half | images/sec  |  64 * N |  563.7913 | 1114.2691 | 2217.8833 | 4386.9452 | 8704.5393 |
efficientnet b4 |  half | hours/epoch |  64 * N |    0.6312 |    0.3194 |    0.1605 |    0.0811 |    0.0409 |


* Throughput (images/sec)

Architecture    |   Batch |   1 GPUs |   2 GPUs |    4 GPUs |    8 GPUs |
----------------|--------:|---------:|---------:|----------:|----------:|
efficientnet b0 | 256 * N | 468.5122 | 903.4220 | 1769.1710 | 3204.8363 |
efficientnet b1 | 192 * N | 288.0718 | 552.2705 | 1079.1396 | 2046.7473 |
efficientnet b2 | 128 * N | 232.7773 | 442.7785 |  857.3406 | 1618.6204 |
efficientnet b3 |  64 * N | 129.7272 | 245.4089 |  471.0875 |  865.3099 |
efficientnet b4 |  32 * N |  59.4473 | 112.2341 |  215.4970 |  381.1867 |
efficientnet b5 |  16 * N |  29.3007 |  53.2217 |  104.1681 |  171.2021 |
efficientnet b6 |  12 * N |  16.9837 |  31.3555 |   60.4685 |  101.2009 |
efficientnet b7 |   8 * N |   9.3913 |  16.5913 |   31.0797 |   42.5191 |

* ETA (hours/epoch = 1281167 / (images/sec) / 60 / 60 = 355.879722222222222 / (images/sec))

Architecture    |   Batch |  1 GPUs |  2 GPUs |  4 GPUs |   8 GPUs |
----------------|--------:|--------:|--------:|--------:|---------:|
efficientnet b0 | 256 * N |  0.7596 |  0.3939 |  0.2012 |  0.11104 |
efficientnet b1 | 192 * N |  1.2353 |  0.6444 |  0.3298 |  0.17387 |
efficientnet b2 | 128 * N |  1.5288 |  0.8037 |  0.4150 |  0.21987 |
efficientnet b3 |  64 * N |  2.7432 |  1.4502 |  0.7554 |  0.41127 |
efficientnet b4 |  32 * N |  5.9865 |  3.1709 |  1.6514 |  0.93361 |
efficientnet b5 |  16 * N | 12.1458 |  6.6867 |  3.4164 |  2.07871 |
efficientnet b6 |  12 * N | 20.9542 | 11.3498 |  6.0744 |  3.51657 |
efficientnet b7 |   8 * N | 37.8946 | 21.4498 | 11.9475 |  8.36988 |


## Optimized Environment

- Python 3.6.9
- PyTorch 1.2.0
- CUDA 10.0.130
- 8 Tesla V100 GPUs
- 8+ Intel E5-2650 v4 CPUs



## To Reproduce

First, resolve the dependencies. We highly recommend to use a separate virtual
environment only for this benchmark:

```bash
$ apt update
$ apt install -y libsm6 libxext-dev libxrender-dev libcap-dev
$ pip install torch torchvision
```

```bash
$ git clone {THIS_REPOSITORY} && cd torchskeleton
$ git submodule init
$ git submodule update
$ pip install -r requirements.txt
```

Prepare ImageNet dataset at `./data/imagenet`:

```sh
$ python -c "import torchvision; torchvision.datasets.ImageNet('./data/imagenet', split='train', download=True)"
$ python -c "import torchvision; torchvision.datasets.ImageNet('./data/imagenet', split='val', download=True)"
```

Then, run each benchmark:
```sh
$ python bin/benchmark/efficientnet_throughput.py -a efficientnet-b0
$ python bin/benchmark/efficientnet_accuracy.py -a efficientnet-b0
```

[paper]: https://arxiv.org/abs/1905.11946
[original]: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
[reproduce]: https://github.com/lukemelas/EfficientNet-PyTorch