import torch
import torch.nn as nn
import torch.utils.data
from torch.nn import functional as F


class Conv_Block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding,
        stride, pool_kernel_size=(2, 2)):
        super(Conv_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size,
            padding, stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size,
            padding, stride)
        self.pool = nn.MaxPool2d(pool_kernel_size)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = self.pool(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 64, 64])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4,
        'padding': 4, 'stride': 1}]
