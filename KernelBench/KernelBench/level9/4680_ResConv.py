import torch
import torch.nn as nn
import torch.utils.data


class ResConv(nn.Module):
    """Some Information about ResConv"""

    def __init__(self, *args, **kwarg):
        super(ResConv, self).__init__()
        self.conv = nn.Conv2d(*args, **kwarg)

    def forward(self, x):
        x = x + self.conv(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
