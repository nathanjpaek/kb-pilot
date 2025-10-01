import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False, groups=groups)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inp, oup, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inp, oup, stride)
        self.norm1 = nn.GroupNorm(2, oup)
        self.conv2 = conv3x3(oup, oup)
        self.norm2 = nn.GroupNorm(2, oup)
        self.relu = nn.ReLU6(inplace=True)
        self.lat = 0
        self.flops = 0
        self.params = 0

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inp': 4, 'oup': 4}]
