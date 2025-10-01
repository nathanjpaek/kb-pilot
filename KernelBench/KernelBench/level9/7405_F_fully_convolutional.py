import torch
import torch.nn as nn
import torch.nn.functional as F


class F_fully_convolutional(nn.Module):

    def __init__(self, in_channels, out_channels, internal_size=256,
        kernel_size=3, leaky_slope=0.02):
        super().__init__()
        pad = kernel_size // 2
        self.leaky_slope = leaky_slope
        self.conv1 = nn.Conv2d(in_channels, internal_size, kernel_size=
            kernel_size, padding=pad)
        self.conv2 = nn.Conv2d(in_channels + internal_size, internal_size,
            kernel_size=kernel_size, padding=pad)
        self.conv3 = nn.Conv2d(in_channels + 2 * internal_size,
            out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x1 = F.leaky_relu(self.conv1(x), self.leaky_slope)
        x2 = F.leaky_relu(self.conv2(torch.cat([x, x1], 1)), self.leaky_slope)
        return self.conv3(torch.cat([x, x1, x2], 1))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
