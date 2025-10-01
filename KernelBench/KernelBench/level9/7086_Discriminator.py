import torch
import torch.nn as nn
from torch.nn import functional as F


class Discriminator(nn.Module):

    def __init__(self, latent_size, d=128):
        super(Discriminator, self).__init__()
        self.latent_size = latent_size
        self.d = d
        self.linear1 = nn.Linear(self.latent_size, self.d)
        self.linear2 = nn.Linear(self.d, self.d)
        self.linear3 = nn.Linear(self.d, 1)

    def forward(self, x):
        x = F.leaky_relu(self.linear1(x), 0.2)
        x = F.leaky_relu(self.linear2(x), 0.2)
        x = torch.sigmoid(self.linear3(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'latent_size': 4}]
