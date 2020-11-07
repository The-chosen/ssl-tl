#!/usr/bin/env bash
nvidia-smi
# conda env create -f ../environment.yml
# source activate yxy
pip install --upgrade torch torchvision  
python train_flower102_yxy.py --transfer-resume '../paths/luna/ssl/1/best.pth.tar' --checkpoint-path '../checkpoints/ssl/flower/1'