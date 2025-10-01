import torch
import torch.nn.parallel
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn
from typing import Optional


def conv3x3(in_channels: 'int', out_channels: 'int', stride: 'int'=1
    ) ->nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=
        stride, padding=1, bias=False)


class ResidualBlock(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', stride:
        'int'=1, downsample: 'Optional[nn.Module]'=None) ->None:
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.downsample = downsample

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
