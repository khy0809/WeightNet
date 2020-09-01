#!/bin/bash
CHECKPOINT="${1}/`date +"%y%m%d"`"
echo $CHECKPOINT

if [ ${TASK_INDEX} == 0 ]; then
  script="mkdir -p ${CHECKPOINT}"
  echo $script
  _=`$script`
fi

script="python -m torch.distributed.launch\
 --nproc_per_node=${TASK_RESOURCE_GPUS}\
 --nnodes=${NUM_TASKS}\
 --master_addr=task1\
 --master_port=2901\
 --node_rank=${TASK_INDEX}\
 bin/benchmark/train_efficientnet_dist.py\
 -c ${CHECKPOINT}/checkpoints\
 --valid-skip 10\
 --seed 0xC0FFEE\
 > ${CHECKPOINT}/log.node${TASK_INDEX}.txt 2>&1 &"
echo $script
eval $script

# bash bin/benchmark/dist_train.sh CHECKOUT_PATH