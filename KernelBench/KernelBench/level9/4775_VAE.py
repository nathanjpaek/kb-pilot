import torch
import torch.nn as nn
import torch.utils.data
from torch.nn import functional as F


class VAE(nn.Module):

    def __init__(self, n_features=24, z_dim=15):
        super(VAE, self).__init__()
        self.en1 = nn.Linear(n_features, 200)
        self.en2 = nn.Linear(200, 100)
        self.en3 = nn.Linear(100, 50)
        self.fc_mu = nn.Linear(50, z_dim)
        self.fc_logvar = nn.Linear(50, z_dim)
        self.de1 = nn.Linear(z_dim, 50)
        self.de2 = nn.Linear(50, 100)
        self.de3 = nn.Linear(100, 200)
        self.de4 = nn.Linear(200, n_features)
        self.n_features = n_features
        self.z_dim = z_dim

    def encode(self, x):
        h1 = F.leaky_relu(self.en1(x))
        h2 = F.leaky_relu(self.en2(h1))
        h3 = F.leaky_relu(self.en3(h2))
        return self.fc_mu(h3), self.fc_logvar(h3)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h4 = F.leaky_relu(self.de1(z))
        h5 = F.leaky_relu(self.de2(h4))
        h6 = F.leaky_relu(self.de3(h5))
        out = self.de4(h6)
        return out

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.n_features))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def get_inputs():
    return [torch.rand([4, 24])]


def get_init_inputs():
    return [[], {}]
