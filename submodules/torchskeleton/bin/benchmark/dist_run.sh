#!/bin/bash
SERVERS=(
"ssh"
"-p 15773 root@gpu-cloud-venode11.dakao.io"
"-p 9134 root@gpu-cloud-venode12.dakao.io"
"-p 17197 root@gpu-cloud-venode13.dakao.io"

"-p 11452 root@gpu-cloud-venode14.dakao.io"
"-p 11118 root@gpu-cloud-venode15.dakao.io"
"-p 24986 root@gpu-cloud-venode16.dakao.io"
"-p 18236 root@gpu-cloud-venode17.dakao.io"

"-p 29520 root@gpu-cloud-venode18.dakao.io"
"-p 29534 root@gpu-cloud-venode19.dakao.io"
"-p 13906 root@gpu-cloud-venode2.dakao.io"
"-p 7652 root@gpu-cloud-venode20.dakao.io"

"-p 8040 root@gpu-cloud-venode8.dakao.io"
"-p 12477 root@gpu-cloud-venode5.dakao.io"
"-p 22045 root@gpu-cloud-venode6.dakao.io"
"-p 8925 root@gpu-cloud-venode7.dakao.io"
)

for server in "${SERVERS[@]}"; do
#  script="ssh ${server} \"pkill python\""
  script="ssh ${server} \"cd /data/private/workspace/torchskeleton; bash bin/benchmark/dist_train.sh runs/efficientnet-b4/batch4k/rmsprop/no-sync-bn\""
  echo ${script}
  eval ${script}
done

# bash bin/benchmark/dist_run.sh