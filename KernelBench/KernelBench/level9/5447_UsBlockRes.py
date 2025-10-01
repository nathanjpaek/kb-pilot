import torch
import torch.nn as nn


def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True):
    return nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=
        stride, padding=padding, bias=bias)


def conv1x1(in_channels, out_channels):
    return nn.Conv3d(in_channels, out_channels, kernel_size=1)


def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2,
            stride=2)
    else:
        return nn.Sequential(nn.Upsample(mode='trilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))


class UsBlockRes(nn.Module):

    def __init__(self, in_channels, out_channels, up_mode='transpose'):
        super(UsBlockRes, self).__init__()
        self.upconv = upconv2x2(in_channels, out_channels, mode=up_mode)
        self.conv = conv3x3(out_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, before_pool, x):
        x = self.upconv(x)
        x = x + before_pool
        x = self.conv(x)
        x = self.relu(x)
        return x


def get_inputs():
    return [torch.rand([4, 8, 8, 8]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
