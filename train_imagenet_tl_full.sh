#!/usr/bin/env bash
nvidia-smi

pip install -r requirements.txt 
pip install --upgrade torch torchvision  
# conda env create -f ../environment.yml
# source activate yxy
# pip install --upgrade torch torchvision  
# python train_pneu_yxy.py --transfer-resume '../paths/imagenet/tl/full/best.pth.tar' --checkpoint-path '../checkpoints_target/imagenet/tl/full/'  --pretrained 'Imagenet'


python train_pneu_yxy.py --transfer-resume '../paths/imagenet/tl/full/best.pth.tar' --checkpoint-path '../checkpoints_target/imagenet/tl/full/1/' --lr 1e-4  --seed 1  --pretrained 'Imagenet'

python train_pneu_yxy.py --transfer-resume '../paths/imagenet/tl/full/best.pth.tar' --checkpoint-path '../checkpoints_target/imagenet/tl/full/10/' --lr 1e-4  --seed 10  --pretrained 'Imagenet'

python train_pneu_yxy.py --transfer-resume '../paths/imagenet/tl/full/best.pth.tar' --checkpoint-path '../checkpoints_target/imagenet/tl/full/1024' --lr 1e-4  --seed 1024  --pretrained 'Imagenet'

# python train_pneu_yxy.py --transfer-resume '../paths/imagenet/tl/full/best.pth.tar' --checkpoint-path '../checkpoints_target/imagenet/tl/full/10/' --lr 1e-4 --seed 10  --pretrained 'Transfer'

# python train_pneu_yxy.py --transfer-resume '../paths/imagenet/tl/full/best.pth.tar' --checkpoint-path '../checkpoints_target/imagenet/tl/full/10/' --lr 5e-4 --seed 10  --pretrained 'Transfer'

# python train_pneu_yxy.py --transfer-resume '../paths/imagenet/tl/full/best.pth.tar' --checkpoint-path '../checkpoints_target/imagenet/tl/full/10/' --lr 1e-3 --seed 10  --pretrained 'Transfer'

# python train_pneu_yxy.py --transfer-resume '../paths/imagenet/tl/full/best.pth.tar' --checkpoint-path '../checkpoints_target/imagenet/tl/full/10/' --lr 1e-5 --seed 10  --pretrained 'Transfer'