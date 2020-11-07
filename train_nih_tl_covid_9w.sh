#!/usr/bin/env bash
nvidia-smi

pip install -r requirements.txt 
pip install --upgrade torch torchvision 
# conda env create -f ../environment.yml
# source activate yxy
pip install --upgrade torch torchvision  
python train_covid_yxy.py --transfer-resume '../paths/nih/tl/9w/best.pth.tar' --checkpoint-path '../checkpoints/tl/covid/9w/1/'  --pretrained 'Transfer'  --seed 1 --logpath '../logs/tl/covid/9w/1/'

python train_covid_yxy.py --transfer-resume '../paths/nih/tl/9w/best.pth.tar' --checkpoint-path '../checkpoints/tl/covid/9w/10/'  --pretrained 'Transfer'  --seed 10 --logpath '../logs/tl/covid/9w/10/'
 
python train_covid_yxy.py --transfer-resume '../paths/nih/tl/9w/best.pth.tar' --checkpoint-path '../checkpoints/tl/covid/9w/1024/'  --pretrained 'Transfer'  --seed 1024 --logpath '../logs/tl/covid/9w/1024/'