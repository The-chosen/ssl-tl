#!/usr/bin/env bash
nvidia-smi
# conda env create -f ../environment.yml
# source activate yxy
pip install --upgrade torch torchvision  
python train_covid_yxy.py --transfer-resume '../paths/sun/ssl/10/best.pth.tar' --checkpoint-path '../checkpoints/ssl/covid/10' 