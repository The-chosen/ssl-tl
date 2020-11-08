#!/bin/sh
pip install -r requirements.txt 
pip install --upgrade torch torchvision   
python train_baseline.py