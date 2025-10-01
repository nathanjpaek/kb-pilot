import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    """A Convolutional Layer"""

    def __init__(self, in_channels=1, out_channels=256, kernel_size=9, stride=1
        ):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        return F.relu(self.conv(x))


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
