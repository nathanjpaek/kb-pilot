import torch
import torch.nn as nn


def conv3x3(in_ch, out_ch, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)


class ResidualBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)
        out = out + identity
        return out


class Enhancement_Block(nn.Module):

    def __init__(self):
        super().__init__()
        self.RB1 = ResidualBlock(32, 32)
        self.RB2 = ResidualBlock(32, 32)
        self.RB3 = ResidualBlock(32, 32)

    def forward(self, x):
        identity = x
        out = self.RB1(x)
        out = self.RB2(out)
        out = self.RB3(out)
        out = out + identity
        return out


def get_inputs():
    return [torch.rand([4, 32, 64, 64])]


def get_init_inputs():
    return [[], {}]
