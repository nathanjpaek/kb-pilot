import torch
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd as autograd
import torch.fft
from itertools import product as product


def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError(
                'sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=
    1, bias=True, mode='CBR', negative_slope=0.2):
    L = []
    for t in mode:
        if t == 'C':
            L.append(nn.Conv2d(in_channels=in_channels, out_channels=
                out_channels, kernel_size=kernel_size, stride=stride,
                padding=padding, bias=bias))
        elif t == 'T':
            L.append(nn.ConvTranspose2d(in_channels=in_channels,
                out_channels=out_channels, kernel_size=kernel_size, stride=
                stride, padding=padding, bias=bias))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=0.0001,
                affine=True))
        elif t == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False)
                )
        elif t == '2':
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == '3':
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == '4':
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == 'U':
            L.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif t == 'u':
            L.append(nn.Upsample(scale_factor=3, mode='nearest'))
        elif t == 'v':
            L.append(nn.Upsample(scale_factor=4, mode='nearest'))
        elif t == 'M':
            L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride,
                padding=0))
        elif t == 'A':
            L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride,
                padding=0))
        else:
            raise NotImplementedError('Undefined type: ')
    return sequential(*L)


class ESA(nn.Module):

    def __init__(self, channel=64, reduction=4, bias=True):
        super(ESA, self).__init__()
        self.r_nc = channel // reduction
        self.conv1 = nn.Conv2d(channel, self.r_nc, kernel_size=1)
        self.conv21 = nn.Conv2d(self.r_nc, self.r_nc, kernel_size=1)
        self.conv2 = nn.Conv2d(self.r_nc, self.r_nc, kernel_size=3, stride=
            2, padding=0)
        self.conv3 = nn.Conv2d(self.r_nc, self.r_nc, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(self.r_nc, self.r_nc, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(self.r_nc, self.r_nc, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(self.r_nc, channel, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = F.max_pool2d(self.conv2(x1), kernel_size=7, stride=3)
        x2 = self.relu(self.conv3(x2))
        x2 = self.relu(self.conv4(x2))
        x2 = F.interpolate(self.conv5(x2), (x.size(2), x.size(3)), mode=
            'bilinear', align_corners=False)
        x2 = self.conv6(x2 + self.conv21(x1))
        return x.mul(self.sigmoid(x2))


class CFRB(nn.Module):

    def __init__(self, in_channels=50, out_channels=50, kernel_size=3,
        stride=1, padding=1, bias=True, mode='CL', d_rate=0.5,
        negative_slope=0.05):
        super(CFRB, self).__init__()
        self.d_nc = int(in_channels * d_rate)
        self.r_nc = in_channels
        assert mode[0] == 'C', 'convolutional layer first'
        self.conv1_d = conv(in_channels, self.d_nc, kernel_size=1, stride=1,
            padding=0, bias=bias, mode=mode[0])
        self.conv1_r = conv(in_channels, self.r_nc, kernel_size, stride,
            padding, bias=bias, mode=mode[0])
        self.conv2_d = conv(self.r_nc, self.d_nc, kernel_size=1, stride=1,
            padding=0, bias=bias, mode=mode[0])
        self.conv2_r = conv(self.r_nc, self.r_nc, kernel_size, stride,
            padding, bias=bias, mode=mode[0])
        self.conv3_d = conv(self.r_nc, self.d_nc, kernel_size=1, stride=1,
            padding=0, bias=bias, mode=mode[0])
        self.conv3_r = conv(self.r_nc, self.r_nc, kernel_size, stride,
            padding, bias=bias, mode=mode[0])
        self.conv4_d = conv(self.r_nc, self.d_nc, kernel_size, stride,
            padding, bias=bias, mode=mode[0])
        self.conv1x1 = conv(self.d_nc * 4, out_channels, kernel_size=1,
            stride=1, padding=0, bias=bias, mode=mode[0])
        self.act = conv(mode=mode[-1], negative_slope=negative_slope)
        self.esa = ESA(in_channels, reduction=4, bias=True)

    def forward(self, x):
        d1 = self.conv1_d(x)
        x = self.act(self.conv1_r(x) + x)
        d2 = self.conv2_d(x)
        x = self.act(self.conv2_r(x) + x)
        d3 = self.conv3_d(x)
        x = self.act(self.conv3_r(x) + x)
        x = self.conv4_d(x)
        x = self.act(torch.cat([d1, d2, d3, x], dim=1))
        x = self.esa(self.conv1x1(x))
        return x


def get_inputs():
    return [torch.rand([4, 50, 64, 64])]


def get_init_inputs():
    return [[], {}]
