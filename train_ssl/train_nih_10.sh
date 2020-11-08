#!/usr/bin/env bash
pip install -r requirements.txt 
pip install --upgrade torch torchvision   

python main_moco_source_NIH.py -b 128 --save-path './checkpoints/checkpoint_resent18_10' --root-dir '../../../../../datasets/nih/images_10' --moco-k 6400 
