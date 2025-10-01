import torch
import torch.nn as nn


def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True):
    return nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=
        stride, padding=padding, bias=bias)


class DsBlock(nn.Module):

    def __init__(self, in_channels, out_channels, pooling):
        super(DsBlock, self).__init__()
        self.conv = conv3x3(in_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pooling = pooling
        if pooling:
            self.mp = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        before_pool = out
        if self.pooling:
            out = self.mp(out)
        return out, before_pool


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'pooling': 4}]
