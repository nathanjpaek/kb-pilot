import torch
from torch import nn


class LayerNorm(nn.Module):

    def __init__(self, size, eps=1e-06):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(size, 1, 1))
        self.bias = nn.Parameter(torch.zeros(size, 1, 1))

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        std = x.std(1, keepdim=True)
        normed = (x - mean) / (std + self.eps)
        return self.weight * normed + self.bias


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'size': 4}]
