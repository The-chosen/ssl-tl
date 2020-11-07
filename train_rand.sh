#!/usr/bin/env bash
nvidia-smi
conda env create -f ../environment.yml
source activate yxy
python train_pneu.py --checkpoint-path '../checkpoints_target/rand'  --pretrained 'None'