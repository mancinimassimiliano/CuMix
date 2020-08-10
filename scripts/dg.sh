#!/bin/bash

#DG=('photo' 'art_painting' 'cartoon' 'sketch')
DG=('cartoon' 'sketch')
DATA=$1
for target in ${DG[@]} ; do
    python -m torch.distributed.launch --nproc_per_node=1 main.py --dg --target $target --config_file configs/dg/dg.json --data_root $DATA --name $target
done
