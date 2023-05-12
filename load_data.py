# -*- coding: utf-8 -*-
"""
Created on Fri May 12 11:52:16 2023

@author: CEOSpaceTech
"""
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from skimage.io import imread

def download(data_path,Categories):
      #################### CNN torch ###################### 
    Categories = os.listdir(data_path)
    flat_data_arr=[] #input array 
    target_arr=[] #output array
    datadir= data_path# +'/Train/' 
    #path which contains all the categories of images
    for i in Categories:
        
        print(f'loading... category : {i}')
        path=os.path.join(datadir,i)
        for img in os.listdir(path):
            img_array=imread(os.path.join(path,img))
            img_cropped=img_array[111:367,111:367]
            # img_resized=resize(img_array,(64,64)) # resize change the pixel values
            flat_data_arr.append(img_cropped) # if its RGB u should use .transpose(2,0,1) to move the num channel first
            target_arr.append(Categories.index(i))
        print(f'loaded category:{i} successfully')
    x=np.array(flat_data_arr)
    y=np.array(target_arr)
    print('Data:', x.shape, y.shape)
    return (x, y)

data_path = 'D:/Omid/UPB/Dataset/GeoTIFF'
(x, y) = download(data_path,'F')
# np.save('F_crop.npy',x)
# np.save('y.npy',y)
