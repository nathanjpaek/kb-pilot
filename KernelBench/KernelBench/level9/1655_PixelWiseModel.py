import torch
from torch import nn as nn


class PixelWiseModel(nn.Module):
    """
    Baseline class for pixelwise models
    Args:
    """

    def __init__(self, const, **kwargs):
        super(PixelWiseModel, self).__init__()
        self.const = const

    def forward(self, x):
        ret = torch.zeros_like(x)
        ret[:] = self.const
        return ret


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'const': 4}]
