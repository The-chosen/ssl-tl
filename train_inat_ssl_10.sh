#!/usr/bin/env bash
nvidia-smi
pip install -r requirements.txt 
pip install --upgrade torch torchvision  

# conda env create -f ../environment.yml
# source activate yxy
# python train_pneu.py --transfer-resume '../paths/inat/ssl/10/best.pth.tar' --checkpoint-path '../checkpoints_target/inat/ssl/10/'  --pretrained 'MoCo'

# python train_pneu_yxy.py --seed 1 --transfer-resume '../paths/inat/ssl/10/best.pth.tar' --checkpoint-path '../checkpoints_target/inat/ssl/10/1/'  --pretrained 'MoCo'

# python train_pneu_yxy.py --seed 10 --transfer-resume '../paths/inat/ssl/10/best.pth.tar' --checkpoint-path '../checkpoints_target/inat/ssl/10/10/'  --pretrained 'MoCo'

# python train_pneu_yxy.py --seed 1024 --transfer-resume '../paths/inat/ssl/10/best.pth.tar' --checkpoint-path '../checkpoints_target/inat/ssl/10/1024/'  --pretrained 'MoCo'

python train_pneu_yxy.py --seed 1 --transfer-resume '../paths/inat/ssl/10/best.pth.tar' --checkpoint-path '../checkpoints_target/inat/ssl/10/1/' --lr 1e-3 --pretrained 'MoCo'

python train_pneu_yxy.py --seed 1 --transfer-resume '../paths/inat/ssl/10/best.pth.tar' --checkpoint-path '../checkpoints_target/inat/ssl/10/1/' --lr 5e-4 --pretrained 'MoCo'

python train_pneu_yxy.py --seed 1 --transfer-resume '../paths/inat/ssl/10/best.pth.tar' --checkpoint-path '../checkpoints_target/inat/ssl/10/1/' --lr 5e-5 --pretrained 'MoCo'

python train_pneu_yxy.py --seed 1 --transfer-resume '../paths/inat/ssl/10/best.pth.tar' --checkpoint-path '../checkpoints_target/inat/ssl/10/1/' --lr 1e-5 --pretrained 'MoCo'