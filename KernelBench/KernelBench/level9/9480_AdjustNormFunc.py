import torch
import torch.nn as nn


class AdjustNormFunc(nn.Module):
    """Creates a BatchNorm-like module using func : x = func(x) * scale + shift"""

    def __init__(self, nf, func=torch.tanh, name=None):
        super().__init__()
        self.func = func
        self.name = name
        self.nf = nf
        self.scale = nn.Parameter(torch.ones(nf, 1, 1))
        self.shift = nn.Parameter(torch.zeros(nf, 1, 1))

    def forward(self, x):
        x = self.func(x)
        return x * self.scale + self.shift

    def __str__(self):
        if self.name:
            return 'Adjusted ' + self.name + f'({self.nf})'
        return 'Adjusted ' + self.func.__str__() + f'({self.nf})'

    def __repr__(self):
        return self.__str__()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nf': 4}]
