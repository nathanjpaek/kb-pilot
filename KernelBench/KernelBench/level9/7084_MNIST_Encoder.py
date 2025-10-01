import torch
import torch.nn as nn
from torch.nn import functional as F


class MNIST_Encoder(nn.Module):

    def __init__(self, in_channels, latent_size):
        super(MNIST_Encoder, self).__init__()
        self.in_channels = in_channels
        self.latent_size = latent_size
        self.linear1 = nn.Linear(self.in_channels, self.latent_size)
        self.linear2 = nn.Linear(self.latent_size, self.latent_size)

    def forward(self, x):
        x = F.leaky_relu(self.linear1(x), 0.2)
        x = torch.tanh(self.linear2(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'latent_size': 4}]
