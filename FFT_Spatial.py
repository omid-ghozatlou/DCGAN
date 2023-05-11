# -*- coding: utf-8 -*-
"""
Created on Thu May 11 12:40:04 2023

@author: CEOSpaceTech
"""
import argparse
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from skimage.transform import resize
from skimage.io import imread
def download(data_path,Categories):
      #################### CNN torch ###################### 
    flat_data_arr=[] #input array 
    target_arr=[] #output array
    datadir= data_path# +'/Train/' 
    #path which contains all the categories of images
    for i in Categories:
        
        print(f'loading... category : {i}')
        path=os.path.join(datadir,i)
        for img in os.listdir(path):
            img_array=imread(os.path.join(path,img))
            img_resized=resize(img_array,(64,64))
            flat_data_arr.append(img_resized.transpose(2,0,1))
            target_arr.append(Categories.index(i))
        print(f'loaded category:{i} successfully')
    x=np.array(flat_data_arr)
    y=np.array(target_arr)
    print('Data:', x.shape, y.shape)
    return (x, y)
data_path = './data/apple'
# os.makedirs("./imgs/FFT", exist_ok=True)
(x, y) = download(data_path,'a')
X =x[:,1,:,:]

X = (torch.Tensor(X))
fft_pairs = torch.zeros((X.shape[0],2,64, 64))
for i in range(X.shape[0]):
    fft_images = torch.fft.fftn(X[i], dim=(-2, -1))
    mag = (torch.abs(fft_images))
    mag = (mag/mag.max()-0.5)*2
            # print(mag.max())
            # print(mag.min())
    phase = torch.angle(fft_images)

    phase = phase/(3.1416)
            # print(phase.max())
            # print(phase.min())
    fft_pairs[i] = torch.stack((mag , phase), dim=0)
mags =fft_pairs[:,0,:,:]
phases =fft_pairs[:,1,:,:]			
spatial = torch.zeros((fft_pairs.shape[0],64, 64),dtype=torch.float32)
for i in range(fft_pairs.shape[0]):
    phase_sc = (phases[i])*3.1416
    mag_sc = (mags[i]+1)/2
    real = mag_sc * torch.cos(phase_sc)
    imag = mag_sc * torch.sin(phase_sc)
    fourier = torch.view_as_complex(torch.stack((real, imag), dim=-1))

    spatial[i] = torch.fft.ifft2(fourier).real