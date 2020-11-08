#!/usr/bin/env bash
pip install -r requirements.txt 
pip install --upgrade torch torchvision   

python main_moco_source_pneu.py -b 32 --save-path '../../target/tl-ssl/checkpoints/target/pneu/ssl/' --moco-k 6400 
