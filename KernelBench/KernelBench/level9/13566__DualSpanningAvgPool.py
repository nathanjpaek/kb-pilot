import torch
from torch import nn
from typing import *


class _DualSpanningAvgPool(nn.Module):
    """Module with two average pools: one that spans the full height of the image and
    another the spans the full width. Outputs are flattened and concatenated.

    Args:
        rows (int): Number of rows in image.
        cols (int): Number of columns in image.
        reduce_size (int): How man
    """

    def __init__(self, rows, cols, reduce_size=1):
        super().__init__()
        self.pool_h = nn.Sequential(nn.AvgPool2d((rows, reduce_size)), nn.
            Flatten())
        self.pool_w = nn.Sequential(nn.AvgPool2d((reduce_size, cols)), nn.
            Flatten())

    def forward(self, x):
        return torch.cat((self.pool_h(x), self.pool_w(x)), dim=-1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'rows': 4, 'cols': 4}]
