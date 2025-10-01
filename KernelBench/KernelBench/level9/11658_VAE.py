import torch
from torch import nn
import torch.nn.functional as F


class VAE(nn.Module):

    def __init__(self, n_features):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(n_features, 1000)
        self.fc2 = nn.Linear(1000, n_features)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return h1

    def decode(self, z):
        h2 = F.relu(self.fc2(z))
        return h2

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_features': 4}]
