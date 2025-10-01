import torch
import torch.nn as nn
from itertools import product as product


class Normalize(nn.Module):

    def __init__(self, n_channels, scale=1.0):
        super(Normalize, self).__init__()
        self.n_channels = n_channels
        self.scale = scale
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.weight.data *= 0.0
        self.weight.data += self.scale
        self.register_parameter('bias', None)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = x / norm * self.weight.view(1, -1, 1, 1)
        return x

    def __repr__(self):
        return 'Normalize(n_channels=%d, scale=%f)' % (self.n_channels,
            self.scale)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_channels': 4}]
