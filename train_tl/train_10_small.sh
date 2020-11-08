#!/usr/bin/env bash
pip install -r requirements.txt 
pip install --upgrade torch torchvision   
cd ./imbalanced_dataset_sampler
python setup.py install
cd ../
# nvidia-smi
# conda env create -f env.yaml
# source activate yy
python train.py  --logpath ./logs/log_10 --checkpoint-path ./checkpoints/checkpoint_10 --train-dir ../../../../datasets/mr_ct_raw_dataset/luna_new_tl_10/train --val-dir  ../../../../datasets/mr_ct_raw_dataset/luna_new_tl_10/val
