import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-08):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        _norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / _norm.clamp(min=self.eps) * self.g


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
