#!/bin/bash
ROOT=$1
mkdir -p $ROOT
cd $ROOT

# Data from here: https://drive.google.com/drive/folders/0B6x7gtvErXgfUU1WcGY5SzdwZVk
# Commands from here: https://medium.com/@acpanjan/download-google-drive-files-using-wget-3c2c025a8b99

# Download raw data
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=0B6x7gtvErXgfbF9CSk53UkRxVzg' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0B6x7gtvErXgfbF9CSk53UkRxVzg" -O pacs.zip && rm -rf /tmp/cookies.txt
unzip pacs.zip
rm pacs.zip

# Download splits
wget 'https://drive.google.com/uc?export=download&id=1RyCVv67zVITU4JYYr1XxLnG5q165yu8s' -O photo_train_kfold.txt
wget 'https://drive.google.com/uc?export=download&id=1lXtOcPL-yWli_MuDZ0mQwLfq0XbzDu--' -O photo_crossval_kfold.txt
wget 'https://drive.google.com/uc?export=download&id=1opW_IbwDQQmuylquX5UvRFxbj3muP5sz' -O photo_test_kfold.txt

wget 'https://drive.google.com/uc?export=download&id=1EhlIyRoeZ5UPiUJs_UXGH-Ql1PS5qtYW' -O cartoon_train_kfold.txt
wget 'https://drive.google.com/uc?export=download&id=1suUUrxyPOmN_dAVdhVSaG-NZDMOsl8JR' -O cartoon_crossval_kfold.txt
wget 'https://drive.google.com/uc?export=download&id=14fpMKzcherr1onzp8jcyhmjJMjZSZ72h' -O cartoon_test_kfold.txt

wget 'https://drive.google.com/uc?export=download&id=1cSo_5_cQLKoURsWwsKqFmw9neWFvDIeV' -O art_painting_train_kfold.txt
wget 'https://drive.google.com/uc?export=download&id=1g7haGuvxmZA1SxPyGFZneDVi9wOm3nK4' -O art_painting_crossval_kfold.txt
wget 'https://drive.google.com/uc?export=download&id=1KVe_S73jECKG9MsIujF-yqZAjPrf0y_O' -O art_painting_test_kfold.txt

wget 'https://drive.google.com/uc?export=download&id=13rWm64rW_dcdcyTFugu4fJcSCTkCB8YA' -O sketch_train_kfold.txt
wget 'https://drive.google.com/uc?export=download&id=1Rs0dUTduQ320bXwHBLQT1s82gyr92J9J' -O sketch_crossval_kfold.txt
wget 'https://drive.google.com/uc?export=download&id=1dFnnsWxYHvdCXmWdhPVwuJ8zgkb_wtbQ' -O sketch_test_kfold.txt

