import torch
from typing import Union
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [(x // 2) for x in k]
    return p


class ConvTranspose(nn.Module):

    def __init__(self, input_channels: 'int', output_channels: 'int',
        kernel_size: 'Tuple[int, int]'=(2, 2), stride: 'Tuple[int, int]'=(2,
        2), padding: 'Union[int, None]'=None, **kwargs):
        super(ConvTranspose, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.transpose = nn.ConvTranspose2d(in_channels=self.input_channels,
            out_channels=self.output_channels, kernel_size=kernel_size,
            stride=stride, padding=autopad(k=kernel_size, p=padding), **kwargs)

    def forward(self, x1: 'torch.Tensor', x2: 'torch.Tensor') ->torch.Tensor:
        out = self.transpose(x1)
        diffY = x2.size()[2] - out.size()[2]
        diffX = x2.size()[3] - out.size()[3]
        out = F.pad(out, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY -
            diffY // 2])
        out = torch.cat([x2, out], dim=1)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_channels': 4, 'output_channels': 4}]
