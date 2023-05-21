# -*- coding: utf-8 -*-
"""
Created on Wed May 17 14:37:24 2023

@author: CEOSpaceTech
"""
import torch
import torch_dct as dct
import os
import numpy as np
import PIL.Image as pilimg
from PIL import Image


###################  Kagle DCT
# from skimage.io import imread_collection
# from torchvision.utils import save_image
# import torchvision.transforms.functional as TF
# import torch_dct as dct

# coldir='D:/Omid/UPB/Dataset/GeoTIFF/data_k/00001/img00001000.png'
# col = imread_collection(coldir)
# print(col)

# for i in range(0, len(col)):
#     dct2= dct.dct_2d(TF.to_tensor(col[i]))
#     #print(dct2)
#     save_image(dct2, str(i)+'.jpg', normalize=False)


###############  DCT 

# path='D:/Omid/UPB/Dataset/GeoTIFF/rgb_k/00000/img00000000.png'
path='D:/Omid/UPB/Dataset/GeoTIFF/K/s1a-wv1-slc-vv-20160101t004844-20160101t004847-009293-00d6be-001.tiff'
# path ='D:/Omid/UPB/GAN/data/apple/a/n07740461_198.jpg'
img=np.array(Image.open(path))
x = (torch.Tensor(img))
# path='D:/Omid/UPB/Dataset/GeoTIFF/K/s1a-wv1-slc-vv-20160629t134811-20160629t134814-011926-0125f4-103.tiff'
# img=np.array(Image.open(path))
# x = (torch.Tensor(img))

X = dct.dct_2d(x, norm='ortho')   # DCT-II done through the last dimension
y = dct.idct_2d(X, norm='ortho')  # scaled DCT-III done through the last dimension
# assert (torch.abs(x - y)).sum() < 1e-10  # x == y within numerical tolerance
X1 = ((X-X.min())/(X.max()-X.min()))*255
magnitude_spectrum = 20 * torch.log(abs(X)+0.01)
# Display the original and reconstructed images side by side
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2)
ax[0].imshow(magnitude_spectrum, cmap='gray')
ax[1].imshow(y, cmap='gray')
plt.show()
