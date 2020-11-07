#!/usr/bin/env bash
nvidia-smi

pip install -r requirements.txt 
pip install --upgrade torch torchvision 
# conda env create -f ../environment.yml
# source activate yxy
pip install --upgrade torch torchvision  

# python train_caltech_yxy.py --transfer-resume '../paths/nih/ssl/9w/best.pth.tar' --checkpoint-path '../checkpoints/ssl/caltech/9w/1/'  --pretrained 'MoCo' --lr 1e-4 --seed 1 --logpath '../logs/ssl/caltech/9w/1/'

# python train_caltech_yxy.py --transfer-resume '../paths/nih/ssl/9w/best.pth.tar' --checkpoint-path '../checkpoints/ssl/caltech/9w/10/'  --pretrained 'MoCo' --lr 1e-4 --seed 10 --logpath '../logs/ssl/caltech/9w/10/'

# python train_caltech_yxy.py --transfer-resume '../paths/nih/ssl/9w/best.pth.tar' --checkpoint-path '../checkpoints/ssl/caltech/9w/1024/'  --pretrained 'MoCo' --lr 1e-4 --seed 1024 --logpath '../logs/ssl/caltech/9w/1024/'

python train_caltech_yxy.py --transfer-resume '../paths/nih/ssl/9w/resnet18/best.pth.tar' --checkpoint-path '../checkpoints/ssl/caltech/9w/1/resnet18/'  --pretrained 'MoCo' --lr 1e-4 --seed 1 --logpath '../logs/ssl/caltech/9w/1/resnet18/' --model-name 'resnet18'

python train_caltech_yxy.py --transfer-resume '../paths/nih/ssl/9w/resnet18/best.pth.tar' --checkpoint-path '../checkpoints/ssl/caltech/9w/10/resnet18/'  --pretrained 'MoCo' --lr 1e-4 --seed 10 --logpath '../logs/ssl/caltech/9w/10/resnet18/' --model-name 'resnet18'

python train_caltech_yxy.py --transfer-resume '../paths/nih/ssl/9w/resnet18/best.pth.tar' --checkpoint-path '../checkpoints/ssl/caltech/9w/1024/resnet18/'  --pretrained 'MoCo' --lr 1e-4 --seed 1024 --logpath '../logs/ssl/caltech/9w/1024/resnet18/' --model-name 'resnet18'


# tune

# python train_caltech_yxy.py --transfer-resume '../paths/nih/ssl/9w/best.pth.tar' --checkpoint-path '../checkpoints/ssl/caltech/9w/1/'  --pretrained 'MoCo' --lr 1e-4 --seed 1 --logpath '../logs/ssl/caltech/9w/1/'

# python train_caltech_yxy.py --transfer-resume '../paths/nih/ssl/9w/best.pth.tar' --checkpoint-path '../checkpoints/ssl/caltech/9w/1/'  --pretrained 'MoCo' --lr 1e-3 --seed 1 --logpath '../logs/ssl/caltech/9w/1/'

# python train_caltech_yxy.py --transfer-resume '../paths/nih/ssl/9w/best.pth.tar' --checkpoint-path '../checkpoints/ssl/caltech/9w/1/'  --pretrained 'MoCo' --lr 5e-5 --seed 1 --logpath '../logs/ssl/caltech/9w/1/'