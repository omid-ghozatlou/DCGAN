# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 18:12:47 2022

@author: CEOSpace
"""

import argparse
import os
import numpy as np
import math

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
# import tifffile

os.makedirs("imgs", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=2, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=50, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

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
os.makedirs("./imgs/FFT", exist_ok=True)
(X, y) = download(data_path,'a')
X =X[:,1,:,:]
# X =np.squee
# X = np.load('F.npy')
train_data = (torch.Tensor(X))
dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle = True, num_workers = 0, pin_memory = True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))
# Transform each image in the batch into the Fourier domain
        fft_pairs = torch.zeros((imgs.shape[0],2, opt.img_size, opt.img_size))
        for i in range(imgs.shape[0]):
            fft_images = torch.fft.fftn(real_imgs[i], dim=(-2, -1))
            mag = (torch.abs(fft_images))
            mag = (mag/mag.max()-0.5)*2
            # print(mag.max())
            # print(mag.min())
            phase = torch.angle(fft_images)
            phase = phase/(3.1416)
            # print(phase.max())
            # print(phase.min())
            fft_pairs[i] = torch.stack((mag , phase), dim=0)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)
        mags =gen_imgs.detach()[:,0,:,:]
        phases =gen_imgs.detach()[:,1,:,:]
        # print(gen_imgs[0].shape)
        # print(gen_imgs.dtype)
        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
		# inverse trandform to spatial domain
        spatial = torch.zeros((gen_imgs.shape[0], opt.img_size, opt.img_size),dtype=torch.float32)
        for i in range(gen_imgs.shape[0]):
            phase_sc = (phases[i])*3.1416
            mag_sc = (mags[i]+1)/2
            real = mag_sc * torch.cos(phase_sc)
            imag = mag_sc * torch.sin(phase_sc)
            fourier = torch.view_as_complex(torch.stack((real, imag), dim=-1))

            spatial[i] = torch.fft.ifft2(fourier).real
        spatial = ((spatial+1)/2)
        # print(spatial.min())
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(fft_pairs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

    save_image(torch.unsqueeze(spatial.data[:25],1), "imgs/FFT/%d.png" % epoch, nrow=5, normalize=True)
        # batches_done = epoch * len(dataloader) + i
        # if batches_done % opt.sample_interval == 0:
        #     save_image(spatial.data[:25], "imgs/FFT/%d.png" % batches_done, nrow=5, normalize=True)