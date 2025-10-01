import torch
import torch.nn as nn
from typing import *


class RegModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.a, self.b = nn.Parameter(torch.randn(1)), nn.Parameter(torch.
            randn(1))

    def forward(self, x):
        return x * self.a + self.b


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
