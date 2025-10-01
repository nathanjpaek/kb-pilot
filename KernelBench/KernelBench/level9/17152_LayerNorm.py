import torch
from torch import nn
from typing import *


class LayerNorm(nn.Module):
    """Normalize by channels, height and width for images."""
    __constants__ = ['eps']

    def __init__(self, eps):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean((1, 2, 3), keepdim=True)
        var = x.var((1, 2, 3), keepdim=True)
        x = (x - mean) / (var + self.eps).sqrt()
        return self.gamma * x + self.beta


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'eps': 4}]
