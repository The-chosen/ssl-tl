#!/usr/bin/env bash
nvidia-smi

pip install -r requirements.txt 
pip install --upgrade torch torchvision  

# conda env create -f ../environment.yml
# source activate yxy
# python train_pneu.py --transfer-resume '../paths/imagenet/tl/1/best.pth.tar' --checkpoint-path '../checkpoints_target/imagenet/tl/1/'  --pretrained 'Transfer'

python train_pneu_yxy.py --transfer-resume '../paths/imagenet/tl/1/best.pth.tar' --checkpoint-path '../checkpoints_target/imagenet/tl/1/1/'  --seed 1  --pretrained 'Transfer'

python train_pneu_yxy.py --transfer-resume '../paths/imagenet/tl/1/best.pth.tar' --checkpoint-path '../checkpoints_target/imagenet/tl/1/10/'  --seed 10  --pretrained 'Transfer'

python train_pneu_yxy.py --transfer-resume '../paths/imagenet/tl/1/best.pth.tar' --checkpoint-path '../checkpoints_target/imagenet/tl/1/1024'  --seed 1024  --pretrained 'Transfer'

