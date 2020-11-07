#!/usr/bin/env bash
nvidia-smi
# conda env create -f ../environment.yml
# source activate yxy
python train_flower102_yxy.py --transfer-resume '../../../tl/checkpoints/checkpoint_1_new/best.pth.tar' --checkpoint-path '../checkpoints/tl/flower/1'