import torch
import torch.nn as nn
import torch.optim


class EnDown(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(EnDown, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3,
            stride=2, padding=1)

    def forward(self, x):
        y = self.conv(x)
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
