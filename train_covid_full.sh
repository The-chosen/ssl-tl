#!/usr/bin/env bash
nvidia-smi
# conda env create -f ../environment.yml
# source activate yxy
pip install --upgrade torch torchvision  
python train_covid_yxy.py --transfer-resume '../../../ssl/moco/checkpoints/checkpoint1/checkpoint_0580.pth.tar' --checkpoint-path '../checkpoints/ssl/covid/full' 