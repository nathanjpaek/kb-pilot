import torch
from torch import Tensor
from torch import nn
from typing import *


class VariableSoftmax(nn.Softmax):
    """Softmax with temperature"""

    def __init__(self, temp: 'float'=1, dim: 'int'=-1):
        super().__init__(dim=dim)
        self.temp = temp

    def forward(self, x: 'Tensor') ->Tensor:
        return super().forward(x / self.temp)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
