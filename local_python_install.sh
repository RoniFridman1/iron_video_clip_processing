#! /bin/bash
set -e

python3 -m virtualenv ./venv
source ./venv/bin/activate
pip install -r ./requirements.txt
# Download models for ironvideo
python3 -c 'from ironvideo import clip_processor; clip_processor.MainLogic()'

# Install CodeFormer
cd ./CodeFormer
pip install -r requirements.txt
python3 basicsr/setup.py develop
cd ..
python3 CodeFormer/scripts/download_pretrained_models.py all

# YOLO tracking stuff
pip install -e ./yolo_tracking/
