#!/usr/bin/env bash
nvidia-smi
# conda env create -f ../environment.yml
# source activate yxy
python train_flower102_yxy.py --transfer-resume '../../../tl/checkpoints/checkpoint_10/best.pth.tar' --checkpoint-path '../checkpoints/tl/flower/10'