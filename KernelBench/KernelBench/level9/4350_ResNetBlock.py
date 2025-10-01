import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int',
        hid_channels: 'int', bias: 'bool'):
        super().__init__()
        self.shortcut = in_channels != out_channels
        self.conv_0 = nn.Conv2d(in_channels, hid_channels, 3, stride=1,
            padding=1)
        self.conv_1 = nn.Conv2d(hid_channels, out_channels, 3, stride=1,
            padding=1, bias=bias)
        if self.shortcut:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 1,
                stride=1, bias=False)

    def forward(self, x):
        xs = x if not self.shortcut else self.conv_shortcut(x)
        x = F.leaky_relu(self.conv_0(x), 0.2)
        x = F.leaky_relu(self.conv_1(x), 0.2)
        return xs + 0.1 * x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'hid_channels': 4,
        'bias': 4}]
