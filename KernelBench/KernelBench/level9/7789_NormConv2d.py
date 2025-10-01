import torch
from torch import nn
from torch.nn.utils import weight_norm


class NormConv2d(nn.Module):
    """
    Convolutional layer with l2 weight normalization and learned scaling parameters
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros([1, out_channels, 1, 1], dtype
            =torch.float32))
        self.gamma = nn.Parameter(torch.ones([1, out_channels, 1, 1], dtype
            =torch.float32))
        self.conv = weight_norm(nn.Conv2d(in_channels, out_channels,
            kernel_size, stride, padding), name='weight')

    def forward(self, x):
        out = self.conv(x)
        out = self.gamma * out + self.beta
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
