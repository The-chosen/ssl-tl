#!/usr/bin/env bash
nvidia-smi
pip install -r requirements.txt 
# conda env create -f ../environment.yml
# source activate yxy
pip install --upgrade torch torchvision  
# python train_flower102_yxy.py --transfer-resume '../paths/nih/tl/9k/best.pth.tar' --checkpoint-path '../checkpoints/tl/flower/9k/'  --pretrained 'Transfer'

python train_flower102_yxy.py --transfer-resume '../paths/nih/tl/10/resnet18/best.pth.tar' --checkpoint-path '../checkpoints/tl/flower/9k/1/resnet18'  --pretrained 'Transfer'  --seed 1 --logpath '../logs/tl/flower/9k/1/'  --lr 1e-3 --model-name 'resnet18'

python train_flower102_yxy.py --transfer-resume '../paths/nih/tl/10/resnet18/best.pth.tar' --checkpoint-path '../checkpoints/tl/flower/9k/10/resnet18'  --pretrained 'Transfer'  --seed 10 --logpath '../logs/tl/flower/9k/10/'  --lr 1e-3 --model-name 'resnet18'

python train_flower102_yxy.py --transfer-resume '../paths/nih/tl/10/resnet18/best.pth.tar' --checkpoint-path '../checkpoints/tl/flower/9k/1024/resnet18'  --pretrained 'Transfer'  --seed 1024 --logpath '../logs/tl/flower/9k/1024/'  --lr 1e-3 --model-name 'resnet18'

# Tune

# python train_flower102_yxy.py --transfer-resume '../paths/nih/tl/10/best.pth.tar' --checkpoint-path '../checkpoints/tl/flower/9k/1/'  --pretrained 'Transfer'  --seed 1 --lr 1e-3 --logpath '../logs/tl/flower/9k/1/'

# python train_flower102_yxy.py --transfer-resume '../paths/nih/tl/10/best.pth.tar' --checkpoint-path '../checkpoints/tl/flower/9k/1/'  --pretrained 'Transfer'  --seed 1 --lr 5e-4 --logpath '../logs/tl/flower/9k/1/'

# python train_flower102_yxy.py --transfer-resume '../paths/nih/tl/10/best.pth.tar' --checkpoint-path '../checkpoints/tl/flower/9k/1/'  --pretrained 'Transfer'  --seed 1 --lr 5e-5 --logpath '../logs/tl/flower/9k/1/'