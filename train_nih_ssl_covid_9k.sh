#!/usr/bin/env bash
nvidia-smi
# conda env create -f ../environment.yml
# source activate yxy
pip install --upgrade torch torchvision  
python train_covid_yxy.py --transfer-resume '../paths/nih/ssl/9k/best.pth.tar' --checkpoint-path '../checkpoints/ssl/covid/9k'  --pretrained 'MoCo' --lr 5e-4