import torch
from torch import Tensor
from typing import Tuple
from typing import Union
import torch.nn as nn
import torch.nn.functional as F


class Conv2dWithFastWeight(nn.Conv2d):

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size:
        'Union[int, Tuple]', stride: 'Union[int, Tuple]'=1, padding:
        'Union[int, Tuple, str]'=0, bias: 'bool'=True) ->None:
        super().__init__(in_channels, out_channels, kernel_size, stride=
            stride, padding=padding, bias=bias)
        self.weight.fast = None
        if self.bias is not None:
            self.bias.fast = None

    def forward(self, x: 'Tensor') ->Tensor:
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None, stride=self.
                    stride, padding=self.padding)
            else:
                out = super().forward(x)
        elif self.weight.fast is not None and self.bias.fast is not None:
            out = F.conv2d(x, self.weight.fast, self.bias.fast, stride=self
                .stride, padding=self.padding)
        else:
            out = super().forward(x)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
