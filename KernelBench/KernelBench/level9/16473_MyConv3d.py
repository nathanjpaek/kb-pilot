import torch
import torch.nn as nn
import torch.nn.functional as F


class MyConv3d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        bias=True):
        super(MyConv3d, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=
            out_channels, kernel_size=kernel_size, padding=(0, int((
            kernel_size - 1) / 2), int((kernel_size - 1) / 2)), bias=bias)

    def forward(self, x):
        x = F.pad(x, pad=(0,) * 4 + (int((self.kernel_size - 1) / 2),) * 2,
            mode='replicate')
        return self.conv(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
