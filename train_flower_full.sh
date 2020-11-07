#!/usr/bin/env bash
nvidia-smi
# conda env create -f ../environment.yml
# source activate yxy
python train_flower102_yxy.py --transfer-resume '../../../ssl/moco/checkpoints/checkpoint1/checkpoint_0360.pth.tar' --checkpoint-path '../checkpoints/ssl/flower/full'