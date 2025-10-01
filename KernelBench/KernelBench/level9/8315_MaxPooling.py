import torch
from typing import Union
import torch.nn as nn
from typing import Tuple


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [(x // 2) for x in k]
    return p


class MaxPooling(nn.Module):

    def __init__(self, input_channels: 'int', kernel_size:
        'Tuple[int, int]'=(2, 2), padding: 'Union[int, None]'=None, **kwargs):
        super(MaxPooling, self).__init__()
        self.input_channels = input_channels
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, padding=autopad(k
            =kernel_size, p=padding), **kwargs)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        out = self.pool(x)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_channels': 4}]
