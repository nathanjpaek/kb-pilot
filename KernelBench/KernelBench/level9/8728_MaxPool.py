import torch
import torch.nn as nn


class MaxPool(nn.Module):
    """Module for MaxPool conversion testing
    """

    def __init__(self, inp=10, out=16, kernel_size=3, bias=True):
        super(MaxPool, self).__init__()
        self.conv2d = nn.Conv2d(inp, out, kernel_size=kernel_size, bias=bias)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.pool(x)
        return x


def get_inputs():
    return [torch.rand([4, 10, 64, 64])]


def get_init_inputs():
    return [[], {}]
