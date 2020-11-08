#!/usr/bin/env bash
pip install -r requirements.txt 
pip install --upgrade torch torchvision   
# nvidia-smi
# conda env create -f env.yaml
# source activate yy
python main_moco.py -b 64 --moco-k 6400 --logpath ./logs/log2 --save-path ./checkpoints/checkpoint2 --data ../../../../../datasets/mr_ct_raw_dataset/luna_new_ssl/
