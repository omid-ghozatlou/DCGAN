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
#Define the discriminator network
class Discriminator(nn.Module):
    def init(self):
        super(Discriminator, self).init()
        self.fc = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)
#Define the hyperparameters
batch_size = 128
lr = 0.0002
latent_dim = 100
num_epochs = 200

#Create the dataset and data loader
dataset = FourierTransformDataset(root='./data')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

#Initialize the generator and discriminator networks
generator = Generator(latent_dim)
discriminator = Discriminator()

#Define the loss function and optimizer
criterion = nn.BCELoss()
generator_optimizer = optim.Adam(generator.parameters(), lr=lr)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

#Define the device (CPU or GPU) to use for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Move the generator and discriminator networks to the device
generator = generator.to(device)
discriminator = discriminator.to(device)

#Train the GAN
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader):

        # Train the discriminator
        real_images = data.to(device)
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Generate fake images
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_images = generator(z)

        # Train the discriminator with real and fake images
        discriminator_real_outputs = discriminator(real_images)
        discriminator_fake_outputs = discriminator(fake_images.detach())
        discriminator_real_loss = criterion(discriminator_real_outputs, real_labels)
        discriminator_fake_loss = criterion(discriminator_fake_outputs, fake_labels)
        discriminator_loss = discriminator_real_loss + discriminator_fake_loss

        discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        discriminator_optimizer.step()

        # Train the generator
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_images = generator(z)
        generator_outputs = discriminator(fake_images)
        generator_loss = criterion(generator_outputs, real_labels)

        generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_optimizer.step()

        # Print the loss values every 100 iterations
        if (i+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Discriminator Loss: {discriminator_loss.item():.4f}, Generator Loss: {generator_loss.item():.4f}")

    # Save the generated images every 3 epochs
    if (epoch+1) % 3 == 0:
        with torch.no_grad():
            z = torch.randn(64, latent_dim).to(device)
            fake_images = generator(z).reshape(-1, 3, 256, 256)
            save_image(fake_images, f'generated_images_{epoch+1}.png', nrow=8, normalize=True)
