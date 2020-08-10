#!/bin/bash
ROOT=$1
mkdir -p $ROOT

# Move utils data
cp data/DomainNet/* $ROOT/

cd $ROOT

# Get train/test splits
wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/clipart_train.txt
wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/clipart_test.txt

wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/infograph_train.txt
wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/infograph_test.txt

wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/painting_train.txt
wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/painting_test.txt

wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/quickdraw_train.txt
wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/quickdraw_test.txt

wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/real_train.txt
wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/real_test.txt

wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_train.txt
wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_test.txt

# Get raw data
wget http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip

# Unzip the data
unzip clipart.zip
rm clipart.zip

unzip infograph.zip
rm infograph.zip

unzip painting.zip
rm painting.zip

unzip quickdraw.zip
rm quickdraw.zip

unzip real.zip
rm real.zip

unzip sketch.zip
rm sketch.zip
