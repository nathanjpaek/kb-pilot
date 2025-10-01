import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 5000)
        self.fc2 = nn.Linear(5000, 5000)
        self.fc21 = nn.Linear(5000, 20)
        self.fc22 = nn.Linear(5000, 20)
        self.fc3 = nn.Linear(20, 5000)
        self.fc32 = nn.Linear(5000, 5000)
        self.fc4 = nn.Linear(5000, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h1 = F.relu(self.fc2(h1))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h3 = F.relu(self.fc32(h3))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        mu = mu.detach()
        mu.zero_()
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def get_inputs():
    return [torch.rand([4, 784])]


def get_init_inputs():
    return [[], {}]
