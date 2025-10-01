import torch
import torch.nn as nn
from torch.nn import functional as F


class MNIST_Discriminator(nn.Module):

    def __init__(self, latent_size):
        super(MNIST_Discriminator, self).__init__()
        self.latent_size = latent_size
        self.linear1 = nn.Linear(self.latent_size, self.latent_size // 2)
        self.linear2 = nn.Linear(self.latent_size // 2, 1)

    def forward(self, x):
        x = F.leaky_relu(self.linear1(x), 0.2)
        x = torch.sigmoid(self.linear2(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'latent_size': 4}]
