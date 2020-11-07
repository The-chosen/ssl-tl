#!/usr/bin/env bash
nvidia-smi

pip install -r requirements.txt 
pip install --upgrade torch torchvision 
# conda env create -f ../environment.yml
# source activate yxy
pip install --upgrade torch torchvision   
# python train_pneu_yxy.py --transfer-resume '../paths/nih/tl/9w/best.pth.tar' --checkpoint-path '../checkpoints/tl/pneu/9w/1/'  --pretrained 'Transfer' --lr 5e-4  --seed 1 --logpath '../logs/tl/pneu/9w/1/'

# python train_pneu_yxy.py --transfer-resume '../paths/nih/tl/9w/best.pth.tar' --checkpoint-path '../checkpoints/tl/pneu/9w/10/'  --pretrained 'Transfer' --lr 5e-4  --seed 10 --logpath '../logs/tl/pneu/9w/10/'

# python train_pneu_yxy.py --transfer-resume '../paths/nih/tl/9w/best.pth.tar' --checkpoint-path '../checkpoints/tl/pneu/9w/1024/'  --pretrained 'Transfer' --lr 5e-4  --seed 1024 --logpath '../logs/tl/pneu/9w/1024/'

python train_pneu_yxy.py --transfer-resume '../paths/nih/tl/9w/resnet18/best.pth.tar' --checkpoint-path '../checkpoints/tl/pneu/9w/1/resnet18/'  --pretrained 'Transfer' --lr 5e-4  --seed 1 --logpath '../logs/tl/pneu/9w/1/resnet18/' --model-name 'resnet18'

python train_pneu_yxy.py --transfer-resume '../paths/nih/tl/9w/resnet18/best.pth.tar' --checkpoint-path '../checkpoints/tl/pneu/9w/10/resnet18/'  --pretrained 'Transfer' --lr 5e-4  --seed 10 --logpath '../logs/tl/pneu/9w/10/resnet18/' --model-name 'resnet18'

python train_pneu_yxy.py --transfer-resume '../paths/nih/tl/9w/resnet18/best.pth.tar' --checkpoint-path '../checkpoints/tl/pneu/9w/1024/resnet18/'  --pretrained 'Transfer' --lr 5e-4  --seed 1024 --logpath '../logs/tl/pneu/9w/1024/resnet18/' --model-name 'resnet18'