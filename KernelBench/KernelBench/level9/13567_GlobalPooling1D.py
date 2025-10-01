import torch
from torch import nn
from typing import *


class GlobalPooling1D(nn.Module):

    def __init__(self):
        super(GlobalPooling1D, self).__init__()

    def forward(self, x):
        x = torch.mean(x, 2)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
