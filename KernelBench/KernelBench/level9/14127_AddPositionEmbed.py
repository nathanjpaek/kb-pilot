import torch
import torch.nn as nn
from functools import partial
import torch.utils.cpp_extension


class AddPositionEmbed(nn.Module):

    def __init__(self, size, init_func=partial(nn.init.normal_, std=0.02)):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(size))
        init_func(self.pe)

    def forward(self, x):
        return x + self.pe


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'size': 4}]
