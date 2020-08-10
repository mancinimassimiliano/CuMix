#!/bin/bash

ZSLDG=('clipart' 'infograph' 'painting' 'quickdraw' 'sketch')
DATA=$1
for target in ${ZSLDG[@]} ; do
        python -m torch.distributed.launch --nproc_per_node=1 main.py --zsl --dg --target $target --config_file configs/zsl+dg/$target.json --data_root $DATA --name $target-zsldg
done
