#! /bin/bash
# This file is a setup script to be run on AWS after extracting the files
# from the installer. It should be run from the base directory, and once done
# setting up, will pass all its arguments to the main program.
set -e

BASE_DIR=`pwd`

sudo dnf install -y python3.11 python3.11-devel python3.11-pip libglvnd-glx
sudo yum groupinstall -y 'Development Tools'
sudo yum install -y python3-virtualenv

python3 -m virtualenv -p python3.11 $BASE_DIR/venv
source $BASE_DIR/venv/bin/activate
pip install -r $BASE_DIR/requirements.txt
pip install -r $BASE_DIR/yolo_tracking/requirements.txt
pip install -e $BASE_DIR/yolo_tracking/

cd $BASE_DIR/CodeFormer
pip install -r requirements.txt
python3 basicsr/setup.py develop

cd $BASE_DIR
python3 -m ironvideo.clip_processor --init-only

cd $BASE_DIR
python3 CodeFormer/scripts/download_pretrained_models.py all

python3 -m ironvideo.clip_processor "$@"
