import torch
import torch.nn as nn


def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True):
    return nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=
        stride, padding=padding, bias=bias)


class UsBlock_nounpool(nn.Module):

    def __init__(self, in_channels, out_channels, up_mode='transpose'):
        super(UsBlock_nounpool, self).__init__()
        self.conv = conv3x3(in_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
