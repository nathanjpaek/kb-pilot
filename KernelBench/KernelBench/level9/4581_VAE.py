import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class VAE(nn.Module):

    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.fc1 = nn.Linear(784, 500)
        self.fc21 = nn.Linear(500, self.z_dim)
        self.fc22 = nn.Linear(500, self.z_dim)
        self.fc3 = nn.Linear(self.z_dim, 500)
        self.fc4 = nn.Linear(500, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        mu = self.fc21(h1)
        logvar = self.fc22(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        x = x.view(-1, 784)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def get_inputs():
    return [torch.rand([4, 784])]


def get_init_inputs():
    return [[], {'z_dim': 4}]
