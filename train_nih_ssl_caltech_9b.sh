#!/usr/bin/env bash
nvidia-smi
pip install -r requirements.txt 
# conda env create -f ../environment.yml
# source activate yxy
pip install --upgrade torch torchvision  
# python train_caltech_yxy.py --transfer-resume '../paths/nih/ssl/1/best.pth.tar' --checkpoint-path '../checkpoints/ssl/caltech/9b/'  --pretrained 'MoCo'

python train_caltech_yxy.py --transfer-resume '../paths/nih/ssl/1/resnet18/best.pth.tar' --checkpoint-path '../checkpoints/ssl/caltech/9b/1/resnet18'  --pretrained 'MoCo' --lr 5e-4 --seed 1 --logpath '../logs/ssl/caltech/9b/1/' --model-name 'resnet18'

python train_caltech_yxy.py --transfer-resume '../paths/nih/ssl/1/resnet18/best.pth.tar' --checkpoint-path '../checkpoints/ssl/caltech/9b/10/resnet18'  --pretrained 'MoCo' --lr 5e-4 --seed 10 --logpath '../logs/ssl/caltech/9b/10/' --model-name 'resnet18'

python train_caltech_yxy.py --transfer-resume '../paths/nih/ssl/1/resnet18/best.pth.tar' --checkpoint-path '../checkpoints/ssl/caltech/9b/1024/resnet18'  --pretrained 'MoCo' --lr 5e-4 --seed 1024 --logpath '../logs/ssl/caltech/9b/1024/' --model-name 'resnet18'
