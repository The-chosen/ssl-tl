#!/usr/bin/env bash
nvidia-smi
pip install -r requirements.txt 
# conda env create -f ../environment.yml
# source activate yxy
pip install --upgrade torch torchvision  
# python train_flower102_yxy.py --transfer-resume '../paths/nih/ssl/9k/best.pth.tar' --checkpoint-path '../checkpoints/ssl/flower/9k'  --pretrained 'MoCo' --lr 2e-3

python train_flower102_yxy.py --transfer-resume '../paths/nih/ssl/10/resnet18/best.pth.tar' --checkpoint-path '../checkpoints/ssl/flower/9k/1/resnet18' --lr 5e-5   --pretrained 'MoCo'  --seed 1 --logpath '../logs/ssl/flower/9k/1/' --model-name 'resnet18'

python train_flower102_yxy.py --transfer-resume '../paths/nih/ssl/10/resnet18/best.pth.tar' --checkpoint-path '../checkpoints/ssl/flower/9k/10/resnet18' --lr 5e-5   --pretrained 'MoCo'  --seed 10 --logpath '../logs/ssl/flower/9k/10/' --model-name 'resnet18'

python train_flower102_yxy.py --transfer-resume '../paths/nih/ssl/10/resnet18/best.pth.tar' --checkpoint-path '../checkpoints/ssl/flower/9k/1024/resnet18' --lr 5e-5   --pretrained 'MoCo'  --seed 1024 --logpath '../logs/ssl/flower/9k/1024/' --model-name 'resnet18'

# python train_flower102_yxy.py --transfer-resume '../paths/nih/ssl/10/best.pth.tar' --checkpoint-path '../checkpoints/ssl/flower/9k/1/'  --pretrained 'MoCo'  --seed 1 --lr 1e-3 --logpath '../logs/ssl/flower/9k/1/'

# python train_flower102_yxy.py --transfer-resume '../paths/nih/ssl/10/best.pth.tar' --checkpoint-path '../checkpoints/ssl/flower/9k/1/'  --pretrained 'MoCo'  --seed 1 --lr 5e-4  --logpath '../logs/ssl/flower/9k/1/'

# python train_flower102_yxy.py --transfer-resume '../paths/nih/ssl/10/best.pth.tar' --checkpoint-path '../checkpoints/ssl/flower/9k/1/'  --pretrained 'MoCo'  --seed 1 --lr 5e-5  --logpath '../logs/ssl/flower/9k/1/'
