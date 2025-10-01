import torch
import torch.nn as nn


class Conv_ReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=None, groups=1, bias=True):
        super(Conv_ReLU, self).__init__()
        if padding is None:
            if stride == 1:
                padding = (kernel_size - 1) // 2
            else:
                padding = 0
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=
            out_channels, kernel_size=kernel_size, stride=stride, padding=
            padding, groups=groups, bias=bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        return self.relu(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
