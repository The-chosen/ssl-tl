#!/usr/bin/env bash
pip install -r requirements.txt 
pip install --upgrade torch torchvision   
cd ./imbalanced_dataset_sampler
python setup.py install
cd ../
# nvidia-smi
# conda env create -f env.yaml
# source activate yy
python train_NIH.py  
