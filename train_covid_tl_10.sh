#!/usr/bin/env bash
nvidia-smi
# conda env create -f ../environment.yml
# source activate yxy
pip install --upgrade torch torchvision  
python train_covid_yxy.py --transfer-resume '../../../tl/checkpoints/checkpoint_10/best.pth.tar' --checkpoint-path '../checkpoints/tl/covid/10'