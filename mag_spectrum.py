# -*- coding: utf-8 -*-
"""
Created on Thu May 11 17:24:34 2023

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
 
def fft(image):
    # Convert the image to grayscale and float32 format
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Compute the Fourier Transform of the image
    f = np.fft.fft2(image)

    # Shift the zero frequency component to the center of the spectrum
    fshift = np.fft.fftshift(f)

    # Compute the magnitude spectrum of the Fourier Transform
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    # Normalize the magnitude spectrum to be between 0 and 1
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 1, cv2.NORM_MINMAX)

    # Convert the magnitude spectrum back to a uint8 image
    magnitude_spectrum = (magnitude_spectrum * 255).astype(np.uint8)

    # Convert the magnitude spectrum to RGB format
    # magnitude_spectrum = cv2.cvtColor(magnitude_spectrum, cv2.COLOR_GRAY2RGB)

    # Resize the magnitude spectrum to match the size of the original image
    magnitude_spectrum = cv2.resize(magnitude_spectrum, (image.shape[1], image.shape[0]))

    # Convert the magnitude spectrum to a PyTorch tensor
    magnitude_spectrum = transforms.ToTensor()(magnitude_spectrum)

    return magnitude_spectrum
# apple
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
x_apple =x[:,1,:,:]
magnitude=fft(x_apple[10])
plt.imshow(np.squeeze(magnitude))

#GeoTiff
# X=np.load('./data/GeoTiff.npy')
magnitude=fft(X[0])
plt.imshow(np.squeeze(fft(X[2])))

# plot spectrum of an image file
path='D:/Omid/UPB/Dataset/GeoTIFF/F/s1a-wv1-slc-vv-20160102t205702-20160102t205705-009320-00d784-011.tiff'
img1=np.array(Image.open(path))
img2=img1[111:367,111:367]
magn=fft(img1)
plt.imshow(np.squeeze(magn))

# plot histogram
plt.hist(img2.flatten(), bins=255);plt.show()