#!/usr/bin/env bash
nvidia-smi
# conda env create -f ../environment.yml
# source activate yxy
pip install --upgrade torch torchvision   
python train_pneu_yxy.py --transfer-resume '../paths/luna/tl/full/best.pth.tar' --checkpoint-path '../checkpoints_target/luna/tl/full/'  --pretrained 'Transfer'