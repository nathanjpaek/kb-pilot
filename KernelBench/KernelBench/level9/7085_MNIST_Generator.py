import torch
import torch.nn as nn
from torch.nn import functional as F


class MNIST_Generator(nn.Module):

    def __init__(self, out_channels, latent_size):
        super(MNIST_Generator, self).__init__()
        self.out_channels = out_channels
        self.latent_size = latent_size
        self.linear1 = nn.Linear(self.latent_size, self.out_channels)
        self.linear2 = nn.Linear(self.out_channels, self.out_channels)

    def forward(self, x):
        x = F.leaky_relu(self.linear1(x), 0.2)
        x = F.leaky_relu(self.linear2(x), 0.2)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'out_channels': 4, 'latent_size': 4}]
