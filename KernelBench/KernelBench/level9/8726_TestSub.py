import torch
import torch.nn as nn


class TestSub(nn.Module):
    """Module for Element-wise subtaction conversion testing
    """

    def __init__(self, inp=10, out=16, kernel_size=3, bias=True):
        super(TestSub, self).__init__()
        self.conv2d_1 = nn.Conv2d(inp, out, stride=inp % 3 + 1, kernel_size
            =kernel_size, bias=bias)
        self.conv2d_2 = nn.Conv2d(inp, out, stride=inp % 3 + 1, kernel_size
            =kernel_size, bias=bias)

    def forward(self, x):
        x1 = self.conv2d_1(x)
        x2 = self.conv2d_2(x)
        return x1 - x2


def get_inputs():
    return [torch.rand([4, 10, 64, 64])]


def get_init_inputs():
    return [[], {}]
