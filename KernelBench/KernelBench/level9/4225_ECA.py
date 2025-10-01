import torch
import torch._C
import torch.serialization
from torch import nn
from typing import *


def int_size(x):
    size = tuple(int(s) for s in x.size())
    return size


class ECA(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, in_channels, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(1, 1, kernel_size=k_size, padding=(k_size - 1
            ) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _b, _c, _h, _w = int_size(x)
        y = self.avg_pool(x)
        y = self.conv(y.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        y = self.sigmoid(y)
        return x * y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4}]
