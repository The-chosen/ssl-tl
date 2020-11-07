#!/usr/bin/env bash
nvidia-smi
# conda env create -f ../environment.yml
# source activate yxy
pip install --upgrade torch torchvision  
python train_caltech_yxy.py --transfer-resume '../../../ssl/moco/checkpoints/checkpoint_10/checkpoint_0980.pth.tar' --checkpoint-path '../checkpoints/ssl/caltech/10'