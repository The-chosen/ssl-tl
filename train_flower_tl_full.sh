#!/usr/bin/env bash
nvidia-smi
# conda env create -f ../environment.yml
# source activate yxy
pip install --upgrade torch torchvision  
python train_flower102_yxy.py --transfer-resume '../../../tl/checkpoints/checkpoint1/best.pth.tar' --checkpoint-path '../checkpoints/tl/flower/full' --pretrained 'Transfer' --epoch 300