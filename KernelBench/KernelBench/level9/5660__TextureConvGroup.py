import torch
from torch import nn
from torch.nn import functional as F


def reflect_padding(x, f, s, half=False):
    if half:
        denom = 2
    else:
        denom = 1
    _, _, h, w = x.shape
    pad_w = w * (s / denom - 1) + f - s
    pad_h = h * (s / denom - 1) + f - s
    if pad_w % 2 == 1:
        pad_l = int(pad_w // 2) + 1
        pad_r = int(pad_w // 2)
    else:
        pad_l = pad_r = int(pad_w / 2)
    if pad_h % 2 == 1:
        pad_t = int(pad_h // 2) + 1
        pad_b = int(pad_h // 2)
    else:
        pad_t = pad_b = int(pad_h / 2)
    return F.pad(x, [pad_l, pad_r, pad_t, pad_b], mode='reflect')


class Conv(nn.Module):
    """Convolutional block. 2d-conv -> batch norm -> (optionally) relu"""

    def __init__(self, in_channels, out_channels, kernel, stride=1,
        use_relu=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride)
        self.batch_norm = nn.InstanceNorm2d(out_channels)
        self.use_relu = use_relu

    def forward(self, x):
        h = self.conv(x)
        h = self.batch_norm(h)
        if self.use_relu:
            h = F.relu(h)
        return h


class _TextureConvGroup(nn.Module):
    """Group of 3 convolutional blocks.

    1.- reflect_padding()
    2.- Conv(in_channels, out_channels, kernel=3, use_relu=False)
    3.- LeakyReLU()
    4.- reflect_padding()
    5.- Conv(out_channels, out_channels, kernel=3)
    6.- LeakyReLU()
    7.- reflect_padding()
    8.- Conv(out_channels, out_channels, kernel=1)
    9.- LeakyReLU()
    """

    def __init__(self, in_channels, out_channels):
        super(_TextureConvGroup, self).__init__()
        self.block1 = Conv(in_channels, out_channels, 3, use_relu=False)
        self.block2 = Conv(out_channels, out_channels, 3, use_relu=False)
        self.block3 = Conv(out_channels, out_channels, 1, use_relu=False)

    def forward(self, x):
        h = reflect_padding(x, 3, 1)
        h = self.block1(h)
        h = F.leaky_relu(h)
        h = reflect_padding(h, 3, 1)
        h = self.block2(h)
        h = F.leaky_relu(h)
        h = reflect_padding(h, 1, 1)
        h = self.block3(h)
        h = F.leaky_relu(h)
        return h


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
