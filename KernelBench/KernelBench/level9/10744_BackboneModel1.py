import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from typing import *


class BackboneModel1(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 1, 1)

    def forward(self, x):
        return self.conv1(x)


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
