import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from generative_models_pytorch.mnist import MNIST


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 784)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = torch.tanh(x)
        x = x.view(-1, 1, 28, 28)
        return x


@dataclasses.dataclass
class WassersteinGANTrainingOption:
    d_path: str
    g_path: str
    device: str
    n_critic: int = 5
    epochs: int = 10
    lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999


class WassersteinGAN:
    def __init__(self):
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.data = MNIST()

    def train(self, option: WassersteinGANTrainingOption):
        device = torch.device(option.device)
        self.generator.to(device)
        self.discriminator.to(device)
        loader = self.data.get_loader()
        g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=option.lr,
            betas=(option.beta1, option.beta2),
        )
        d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=option.lr,
            betas=(option.beta1, option.beta2),
        )

        for epoch in range(option.epochs):
            for idx, (X_real, _) in enumerate(loader):
                X_real = X_real.to(device)
                if idx % option.n_critic == 0:
                    g_optimizer.zero_grad()
                    noise = torch.rand(X_real.shape[0], 128, device=device)
                    X_fake = self.generator(noise)
                    y_fake = self.discriminator(X_fake)
                    g_loss = -torch.mean(y_fake)
                    g_loss.backward()
                    g_optimizer.step()

                d_optimizer.zero_grad()
                noise = torch.rand(X_real.shape[0], 128, device=device)
                X_fake = self.generator(noise)
                X_fake = X_fake.detach()
                y_real = self.discriminator(X_real)
                y_fake = self.discriminator(X_fake)
                d_loss = -torch.mean(y_real) + torch.mean(y_fake)
                d_loss.backward()
                d_optimizer.step()
        torch.save(self.generator.state_dict(), option.g_path)
        torch.save(self.discriminator.state_dict(), option.d_path)
