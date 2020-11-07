#!/usr/bin/env bash
nvidia-smi
conda env create -f ../environment.yml
source activate yxy
python train_pneu.py --transfer-resume '../paths/luna/ssl/10/best.pth.tar' --checkpoint-path '../checkpoints_target/luna/ssl/10/'  --pretrained 'MoCo'