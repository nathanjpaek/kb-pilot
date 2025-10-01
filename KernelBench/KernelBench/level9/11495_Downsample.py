import torch
from torch import nn


class Downsample(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_in': 4, 'dim_out': 4}]
