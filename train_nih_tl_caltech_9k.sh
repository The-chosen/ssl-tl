#!/usr/bin/env bash
nvidia-smi
pip install -r requirements.txt 
# conda env create -f ../environment.yml
# source activate yxy
pip install --upgrade torch torchvision  
# python train_caltech_yxy.py --transfer-resume '../paths/nih/tl/9k/best.pth.tar' --checkpoint-path '../checkpoints/tl/caltech/9k/'  --pretrained 'Transfer'

python train_caltech_yxy.py --transfer-resume '../paths/nih/tl/10/resnet18/best.pth.tar' --checkpoint-path '../checkpoints/tl/caltech/9k/1/resnet18'  --pretrained 'Transfer' --lr 1e-3  --seed 1 --logpath '../logs/tl/caltech/9k/1/' --model-name 'resnet18'

python train_caltech_yxy.py --transfer-resume '../paths/nih/tl/10/resnet18/best.pth.tar' --checkpoint-path '../checkpoints/tl/caltech/9k/10/resnet18'  --pretrained 'Transfer'  --lr 1e-3  --seed 10 --logpath '../logs/tl/caltech/9k/10/' --model-name 'resnet18'

python train_caltech_yxy.py --transfer-resume '../paths/nih/tl/10/resnet18/best.pth.tar' --checkpoint-path '../checkpoints/tl/caltech/9k/1024/resnet18'  --pretrained 'Transfer'  --lr 1e-3  --seed 1024 --logpath '../logs/tl/caltech/9k/1024/' --model-name 'resnet18'
