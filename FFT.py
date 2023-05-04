import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import cv2

# Define the Fourier Transform function
def fft(image):
    # Convert the image to grayscale and float32 format
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)

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
    magnitude_spectrum = cv2.cvtColor(magnitude_spectrum, cv2.COLOR_GRAY2RGB)

    # Resize the magnitude spectrum to match the size of the original image
    magnitude_spectrum = cv2.resize(magnitude_spectrum, (image.shape[1], image.shape[0]))

    # Convert the magnitude spectrum to a PyTorch tensor
    magnitude_spectrum = transforms.ToTensor()(magnitude_spectrum)

    return magnitude_spectrum

# Define the custom dataset class that uses Fourier Transform
class FourierTransformDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_files = sorted(glob.glob(os.path.join(root, '*.jpg')))

    def __getitem__(self, index):
        # Load the image
        image = cv2.imread(self.image_files[index])

        # Apply Fourier Transform to the image
        transformed_image = fft(image)

        # Apply the specified transform to the transformed image
        if self.transform:
            transformed_image = self.transform(transformed_image)

        # Return the transformed image
        return transformed_image

    def __len__(self):
        return len(self.image_files)

# Define the generator network
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, z):
        return self.fc(z)

