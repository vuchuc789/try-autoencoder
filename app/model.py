import torch
from torch import nn


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(140, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        self.mean = nn.Linear(16, 8)
        self.logvar = nn.Linear(16, 8)

        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 140),
            nn.Sigmoid(),
        )

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        return mean + std * eps

    def forward(self, x):
        encoded = self.encoder(x)
        mean = self.mean(encoded)
        logvar = self.logvar(encoded)
        latent = self.reparameterize(mean, logvar)
        decoded = self.decoder(latent)

        return decoded, mean, logvar
