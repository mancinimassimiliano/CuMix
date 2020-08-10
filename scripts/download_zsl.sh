#!/bin/bash
ROOT=$1
mkdir -p $ROOT
cd $ROOT

# Download data
wget http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip
wget http://datasets.d2.mpi-inf.mpg.de/xian/cvpr18xian.zip

# Unpack and move datasets
unzip xlsa17.zip
unzip cvpr18xian.zip
mv xlsa17/data/* .
mv cvpr18xian/data/* .

# Remove folders and zips
rm xlsa17.zip
rm cvpr18xian.zip
rm -r cvpr18xian
rm -r xlsa17

# Remove unused datasets
rm -r APY
rm -r AWA2
rm -r ImageNet
