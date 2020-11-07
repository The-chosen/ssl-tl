#!/usr/bin/env bash
nvidia-smi

pip install -r requirements.txt 
pip install --upgrade torch torchvision  
# pip install --upgrade torch torchvision   
# conda env create -f ../environment.yml
# source activate yxy
# python train_pneu_yxy.py --transfer-resume '../paths/sun/ssl/full/best.pth.tar' --checkpoint-path '../checkpoints_target/sun/ssl/full/'  --pretrained 'MoCo'

python train_pneu_yxy.py  --seed 1   --transfer-resume '../paths/sun/ssl/full/best.pth.tar' --checkpoint-path '../checkpoints_target/sun/ssl/full/1/'  --pretrained 'MoCo'

python train_pneu_yxy.py  --seed 10   --transfer-resume '../paths/sun/ssl/full/best.pth.tar' --checkpoint-path '../checkpoints_target/sun/ssl/full/10/'  --pretrained 'MoCo'

python train_pneu_yxy.py  --seed 1024   --transfer-resume '../paths/sun/ssl/full/best.pth.tar' --checkpoint-path '../checkpoints_target/sun/ssl/full/1024/'  --pretrained 'MoCo'