import torch
import torch.nn as nn


class ResidualBlock(nn.Sequential):

    def __init__(self, *args):
        super(ResidualBlock, self).__init__(*args)

    def forward(self, x):
        identity = x
        x = super(ResidualBlock, self).forward(x)
        x += identity
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
