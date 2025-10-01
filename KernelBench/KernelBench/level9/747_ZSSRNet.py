import torch
from torch import nn
import torch.utils.data
import torch


class ZSSRNet(nn.Module):

    def __init__(self, input_channels=3, kernel_size=3, channels=64):
        super(ZSSRNet, self).__init__()
        self.conv0 = nn.Conv2d(input_channels, channels, kernel_size=
            kernel_size, padding=kernel_size // 2, bias=True)
        self.conv7 = nn.Conv2d(channels, input_channels, kernel_size=
            kernel_size, padding=kernel_size // 2, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.relu(self.conv0(x))
        x = self.conv7(x)
        return x + residual


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
