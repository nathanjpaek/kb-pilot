import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.parallel
import torch.optim


class mfm(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
        padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2 * out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2 * out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])


class group(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        padding, mid_channels=None):
        super(group, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels
        self.conv_a = mfm(in_channels, mid_channels, 1, 1, 0)
        self.conv = mfm(mid_channels, out_channels, kernel_size, stride,
            padding)

    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4,
        'stride': 1, 'padding': 4}]
