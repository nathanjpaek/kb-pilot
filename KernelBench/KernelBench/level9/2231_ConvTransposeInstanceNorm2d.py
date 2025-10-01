import torch
import torch.nn as nn
from typing import Tuple
from typing import Union


class ConvTransposeInstanceNorm2d(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size:
        'Union[int, Tuple[int]]', stride: 'Union[int, Tuple[int]]'=1,
        padding: 'Union[int, Tuple[int]]'=0, output_padding:
        'Union[int, Tuple[int]]'=0, dilation: 'Union[int, Tuple[int]]'=1,
        groups: 'int'=1, bias: 'bool'=True, padding_mode: 'str'='zeros'):
        super().__init__()
        self.dconv = nn.ConvTranspose2d(in_channels, out_channels,
            kernel_size, stride, padding, output_padding, groups, bias,
            dilation, padding_mode)
        self.norm = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = self.dconv(x)
        x = self.norm(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
