#!/usr/bin/env bash
nvidia-smi
# conda env create -f ../environment.yml
# source activate yxy
pip install --upgrade torch torchvision  
python train_covid_yxy.py --transfer-resume '../paths/nih/ssl/1/best.pth.tar' --checkpoint-path '../checkpoints/ssl/covid/9b'  --pretrained 'MoCo' --lr 1e-6