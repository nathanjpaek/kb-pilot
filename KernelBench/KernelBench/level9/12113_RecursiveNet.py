import torch
from typing import Any
import torch.nn as nn


class RecursiveNet(nn.Module):
    """ Model that uses a layer recursively in computation. """

    def __init__(self) ->None:
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)

    def forward(self, x: 'torch.Tensor', args1: 'Any'=None, args2: 'Any'=None
        ) ->torch.Tensor:
        del args1, args2
        out = x
        for _ in range(3):
            out = self.conv1(out)
            out = self.conv1(out)
        return out


def get_inputs():
    return [torch.rand([4, 64, 64, 64])]


def get_init_inputs():
    return [[], {}]
