import torch
from torch import Tensor
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.nn.functional as F


class LSN(nn.Module):
    """ Custom Linear layer that modifies standard ReLU layer"""
    __constants__ = ['inplace']
    inplace: 'bool'

    def __init__(self, scale: 'int'=20000, inplace: 'bool'=False):
        super(LSN, self).__init__()
        self.inplace = inplace
        self.scale = scale

    def forward(self, input: 'Tensor') ->Tensor:
        y_relu = F.relu(input, inplace=self.inplace)
        num = y_relu * self.scale
        denom = torch.sum(y_relu)
        return num / (denom + 1e-08)

    def extra_repr(self) ->str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
