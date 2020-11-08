#!/usr/bin/env bash
pip install -r requirements.txt 
pip install --upgrade torch torchvision   
# nvidia-smi
# conda env create -f env.yaml 
# source activate yy
python main_moco.py -b 128 --moco-k 640 --logpath ./logs/log_1 --save-path ./checkpoints/checkpoint_1 --data ../../../../../datasets/mr_ct_raw_dataset/luna_new_ssl_1/
