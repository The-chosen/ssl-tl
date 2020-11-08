#!/usr/bin/env python

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class SUNDataset(data.Dataset):
    def __init__(self, root,number_per_class,number_cls=1000,mode = 'train',transforms=None):
        self.number_cls = number_cls
        self.root = root
        self.category = sorted(os.listdir(root))
        
#         print('Number of sorted class {}'.format(len(self.category)))
        self.images = []
        self.labels = []

        real_category = []
        for i,c in enumerate(self.category):
            cls_dir = os.path.join(self.root,c)
            num_exist = len(os.listdir(cls_dir))
                
            real_category.append(c)

        self.category = real_category[:number_cls]
        
        for i,c in enumerate(self.category):
            cls_dir = os.path.join(self.root,c)
            num_exist = len(os.listdir(cls_dir))
            cls_list = [os.path.join(cls_dir,item) for item in os.listdir(cls_dir)]
            img_list = []
            for i,c in enumerate(cls_list):
                original_number_per_class = len(os.listdir(c))
                if mode=='train':
                    for item in os.listdir(c)[:int(number_per_class*original_number_per_class)]:
                        if os.path.isfile(os.path.join(c, item)):
                            img_list.append(os.path.join(c, item))
                        else:
                            dir = os.path.join(c, item)
                            for subitem in os.listdir(os.path.join(c, item)):         
                                img_list.append(os.path.join(dir, subitem))                                
                else:
                    for item in os.listdir(c)[int(number_per_class*original_number_per_class):]:
                        if os.path.isfile(os.path.join(c, item)):
                            img_list.append(os.path.join(c, item))
                        else:
                            dir = os.path.join(c, item)
                            for subitem in os.listdir(os.path.join(c, item)):          
                                img_list.append(os.path.join(dir, subitem)) 
            self.images.extend(img_list)
            print('Number per class ratio', number_per_class)
            print('Number of images', len(self.images))
            self.labels.extend(len(img_list)*[i])
        self.transforms = transforms
        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        
        img = Image.open(self.images[index]).convert('RGB')
        target = self.labels[index]

        if self.transforms:
            img = self.transforms(img)


        return img, target