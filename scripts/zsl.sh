#!/bin/bash

ZSL=('cub' 'flo' 'awa1' 'sun')
DATA=$1
for target in ${ZSL[@]} ; do
    python -m torch.distributed.launch --nproc_per_node=1 main.py --zsl --target $target --config_file configs/zsl/$target.json --data_root $DATA --name $target --runs 2
done
