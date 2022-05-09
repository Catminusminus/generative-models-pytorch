import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from generative_models_pytorch.mnist import MNIST


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(400, 20)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = F.relu(x)
        mu = self.fc2(x)
        log_var = self.fc3(x)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(20, 400)
        self.fc2 = nn.Linear(400, 784)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


class VAEModel(nn.Module):
    def __init__(self):
        super(VAEModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        y = self.decoder(z)
        return y, mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std


def loss_function(x, y, mu, log_var):
    reconstruction = F.binary_cross_entropy(y, x.view(-1, 784), reduction="sum")
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return reconstruction + KLD


@dataclasses.dataclass
class VAETrainingOption:
    path: str
    device: str
    epochs: int = 10
    lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999


class VAE:
    def __init__(self):
        self.model = VAEModel()
        self.data = MNIST()

    def train(self, option: VAETrainingOption):
        device = torch.device(option.device)
        self.model.to(device)
        loader = self.data.get_loader()
        criterion = loss_function
        optimizer = optim.Adam(
            self.model.parameters(), lr=option.lr, betas=(option.beta1, option.beta2)
        )
        for epoch in range(option.epochs):
            for idex, (X, _) in enumerate(loader):
                X = X.to(device)
                optimizer.zero_grad()
                y, mu, log_var = self.model(X)
                loss = criterion(X, y, mu, log_var)
                loss.backward()
                optimizer.step()
        torch.save(self.model.state_dict(), option.path)
