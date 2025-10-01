import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNeXtBottleneck(nn.Module):

    def __init__(self, in_channels=256, out_channels=256, stride=1,
        cardinality=32, dilate=1):
        super(ResNeXtBottleneck, self).__init__()
        D = out_channels // 2
        self.out_channels = out_channels
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=
            1, padding=0, bias=False)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=2 + stride, stride=
            stride, padding=dilate, dilation=dilate, groups=cardinality,
            bias=False)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride
            =1, padding=0, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut.add_module('shortcut', nn.AvgPool2d(2, stride=2))

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_expand.forward(bottleneck)
        x = self.shortcut.forward(x)
        return x + bottleneck


def get_inputs():
    return [torch.rand([4, 256, 64, 64])]


def get_init_inputs():
    return [[], {}]
