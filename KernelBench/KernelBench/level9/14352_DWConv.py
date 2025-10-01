import torch
from torch import nn


class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
        use_bn=True, use_relu=True, inplace=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=
            out_channels, kernel_size=kernel_size, stride=stride, padding=
            padding, dilation=dilation, groups=groups, bias=bias,
            padding_mode=padding_mode)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else None
        self.relu = nn.ReLU(inplace=inplace) if use_relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class DWConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, use_bn=False, **kwargs):
        super().__init__()
        self.use_bn = use_bn
        self.DWConv = BasicConv(in_channels, in_channels, kernel_size,
            stride, padding, groups=in_channels, use_bn=use_bn)
        self.conv1x1 = BasicConv(in_channels, out_channels, 1, 1, 0, use_bn
            =use_bn)

    def forward(self, x):
        x = self.DWConv(x)
        x = self.conv1x1(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
