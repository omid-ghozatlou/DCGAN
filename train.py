import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.utils import save_image

# Define the hyperparameters
batch_size = 64
latent_dim = 100
num_epochs = 50
learning_rate = 0.0002

# Define the transformations to apply to the images
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the dataset
dataset = ImageFolder(root='path/to/your/dataset', transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the generator network
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 4, 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(-1, self.latent_dim, 1, 1)
        return self.gen(x)

# Define the discriminator network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.dis(x)
        return x.view(-1, 1)

# Initialize the generator and discriminator networks
generator = Generator(latent_dim)
discriminator = Discriminator()

# Define the loss functions and optimizers
adversarial_loss = nn.BCELoss()
generator_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Train the GAN
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(dataloader):
        batch_size = images.shape[0]

        # Train the discriminator
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        
        # Train the discriminator with real images
        discriminator_optimizer.zero_grad()
        real_images = images.cuda()
        real_outputs = discriminator(real_images)
        real_loss = adversarial_loss(real_outputs, real_labels)
        real_loss.backward()

        # Train the discriminator with fake images
        z = torch.randn(batch_size, latent_dim).cuda()
        fake_images = generator(z)
        fake_outputs = discriminator(fake_images.detach())
        fake_loss = adversarial_loss(fake_outputs, fake_labels)
        fake_loss.backward()
        discriminator_optimizer.step()

        # Train the generator
        generator_optimizer.zero_grad()
        z = torch.randn(batch_size, latent_dim).cuda()
        fake_images = generator(z)
        fake_outputs = discriminator(fake_images)
        generator_loss = adversarial_loss(fake_outputs, real_labels)
        generator_loss.backward()
        generator_optimizer.step()

        # Print the loss values
        if i % 100 == 0:
            print('[%d/%d][%d/%d] Real loss: %.4f\tFake loss: %.4f\tGenerator loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(dataloader),
                     real_loss.item(), fake_loss.item(), generator_loss.item()))

        # Save generated images
        if i % 100 == 0:
            with torch.no_grad():
                fake_images = generator(z).detach().cpu()
            save_image(fake_images, 'generated_images/epoch_%03d_batch_%03d.png' % (epoch + 1, i + 1), normalize=True)

    # Save the models
    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')
