#!/usr/bin/env bash
nvidia-smi
# conda env create -f ../environment.yml
# source activate yxy
pip install --upgrade torch torchvision  
python train_covid_yxy.py --transfer-resume '../../../tl/checkpoints/checkpoint1/best.pth.tar' --checkpoint-path '../checkpoints/tl/covid/full'   --pretrained 'Transfer' --lr 4e-4 --epoch 200