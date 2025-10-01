import torch
from torch import nn
import torch.cuda
import torch.backends
import torch.multiprocessing


class GaussianNoise(nn.Module):
    """Add random gaussian noise to images."""

    def __init__(self, std=0.05):
        super(GaussianNoise, self).__init__()
        self.std = std

    def forward(self, x):
        return x + torch.randn(x.size()).type_as(x) * self.std


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
