import torch
from typing import *
from typing import Optional
from torch import nn


class AdaptiveConcatPool2d(nn.Module):
    """Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."""

    def __init__(self, sz: 'Optional[int]'=None):
        super(AdaptiveConcatPool2d, self).__init__()
        """Output will be 2*sz or 2 if sz is None"""
        self.output_size = sz or 1
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, x):
        """
		Compute (1) the maxpool(x), and (2) the averagepool(x), then concatenate their outputs.
		"""
        return torch.cat([self.mp(x), self.ap(x)], 1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
