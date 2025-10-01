import torch
import torch.nn as nn


class TestConv2d(nn.Module):
    """Module for Dense conversion testing
    """

    def __init__(self, inp=10, out=16, kernel_size=3, dilation=1, bias=True):
        super(TestConv2d, self).__init__()
        self.conv2d = nn.Conv2d(inp, out, kernel_size=kernel_size, bias=
            bias, dilation=dilation)

    def forward(self, x):
        x = self.conv2d(x)
        return x


def get_inputs():
    return [torch.rand([4, 10, 64, 64])]


def get_init_inputs():
    return [[], {}]
