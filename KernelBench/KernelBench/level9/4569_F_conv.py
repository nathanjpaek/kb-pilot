import torch
import warnings
import torch.nn as nn
import torch.nn.functional as F
import torch.optim


class F_conv(nn.Module):
    """ResNet transformation, not itself reversible, just used below"""

    def __init__(self, in_channels, channels, channels_hidden=None, stride=
        None, kernel_size=3, leaky_slope=0.1, batch_norm=False):
        super().__init__()
        if stride:
            warnings.warn(
                "Stride doesn't do anything, the argument should be removed",
                DeprecationWarning)
        if not channels_hidden:
            channels_hidden = channels
        pad = kernel_size // 2
        self.leaky_slope = leaky_slope
        self.conv1 = nn.Conv2d(in_channels, channels_hidden, kernel_size=
            kernel_size, padding=pad, bias=not batch_norm)
        self.conv2 = nn.Conv2d(channels_hidden, channels_hidden,
            kernel_size=kernel_size, padding=pad, bias=not batch_norm)
        self.conv3 = nn.Conv2d(channels_hidden, channels, kernel_size=
            kernel_size, padding=pad, bias=not batch_norm)
        if batch_norm:
            self.bn1 = nn.BatchNorm2d(channels_hidden)
            self.bn1.weight.data.fill_(1)
            self.bn2 = nn.BatchNorm2d(channels_hidden)
            self.bn2.weight.data.fill_(1)
            self.bn3 = nn.BatchNorm2d(channels)
            self.bn3.weight.data.fill_(1)
        self.batch_norm = batch_norm

    def forward(self, x):
        out = self.conv1(x)
        if self.batch_norm:
            out = self.bn1(out)
        out = F.leaky_relu(out, self.leaky_slope)
        out = self.conv2(out)
        if self.batch_norm:
            out = self.bn2(out)
        out = F.leaky_relu(out, self.leaky_slope)
        out = self.conv3(out)
        if self.batch_norm:
            out = self.bn3(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'channels': 4}]
