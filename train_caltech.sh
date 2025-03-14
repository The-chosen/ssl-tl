#!/usr/bin/env bash
nvidia-smi
# conda env create -f ../environment.yml
# source activate yxy
pip install --upgrade torch torchvision  
python train_caltech_yxy.py  --transfer-resume '../paths/luna/ssl/full/best.pth.tar' --checkpoint-path '../checkpoints/ssl/caltech/full'