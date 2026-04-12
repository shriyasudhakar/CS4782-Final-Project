# Data download instructions

## Dataset link: https://data.vision.ee.ethz.ch/cvl/DIV2K/
We are using the High Resolution (HR) Train and Validation split. The zips can be found at http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip and http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip

## Download locally in Colab:

train split:
!wget -c http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
!unzip DIV2K_train_HR.zip

valid split:
!wget -c http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
!unzip DIV2K_valid_HR.zip

remove zips after unziping (save some disk space):
!rm -r DIV2K_train_HR.zip
!rm -r DIV2K_valid_HR.zip

## Download onto machine
curl -L -O wget -c http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
unzip DIV2K_train_HR.zip

curl -L -O wget -c http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
unzip DIV2K_valid_HR.zip


