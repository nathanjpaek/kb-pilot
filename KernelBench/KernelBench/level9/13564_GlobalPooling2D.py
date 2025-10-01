import torch
from torch import nn
from typing import *


class GlobalPooling2D(nn.Module):

    def __init__(self):
        super(GlobalPooling2D, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.mean(x, 2)
        x = x.view(x.size(0), -1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
