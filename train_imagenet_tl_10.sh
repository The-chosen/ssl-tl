#!/usr/bin/env bash
nvidia-smi

pip install -r requirements.txt 
pip install --upgrade torch torchvision  

# conda env create -f ../environment.yml
# source activate yxy
# pip install --upgrade torch torchvision  
# python train_pneu_yxy.py --transfer-resume '../paths/imagenet/tl/10/best.pth.tar' --checkpoint-path '../checkpoints_target/imagenet/tl/10/'  --pretrained 'Transfer'


python train_pneu_yxy.py --transfer-resume '../paths/imagenet/tl/10/best.pth.tar' --checkpoint-path '../checkpoints_target/imagenet/tl/10/1/'  --seed 1  --pretrained 'Transfer'

python train_pneu_yxy.py --transfer-resume '../paths/imagenet/tl/10/best.pth.tar' --checkpoint-path '../checkpoints_target/imagenet/tl/10/10/'  --seed 10  --pretrained 'Transfer'

python train_pneu_yxy.py --transfer-resume '../paths/imagenet/tl/10/best.pth.tar' --checkpoint-path '../checkpoints_target/imagenet/tl/10/1024'  --seed 1024  --pretrained 'Transfer'

