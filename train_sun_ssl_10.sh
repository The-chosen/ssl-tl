#!/usr/bin/env bash
nvidia-smi

pip install -r requirements.txt 
pip install --upgrade torch torchvision  
# conda env create -f ../environment.yml
# source activate yxy
# pip install --upgrade torch torchvision  
# python train_pneu_yxy.py --transfer-resume '../paths/sun/ssl/10/best.pth.tar' --checkpoint-path '../checkpoints_target/sun/ssl/10/'  --pretrained 'MoCo'

python train_pneu_yxy.py  --seed 1   --transfer-resume '../paths/sun/ssl/10/best.pth.tar' --checkpoint-path '../checkpoints_target/sun/ssl/10/1/'  --pretrained 'MoCo'

python train_pneu_yxy.py  --seed 10   --transfer-resume '../paths/sun/ssl/10/best.pth.tar' --checkpoint-path '../checkpoints_target/sun/ssl/10/10/'  --pretrained 'MoCo'

python train_pneu_yxy.py  --seed 1024   --transfer-resume '../paths/sun/ssl/10/best.pth.tar' --checkpoint-path '../checkpoints_target/sun/ssl/10/1024/'  --pretrained 'MoCo'