import torch
import torch.nn as nn
import torch.nn.functional as F


class TestUpsampleNearest2d(nn.Module):
    """Module for UpsampleNearest2d conversion testing
    """

    def __init__(self, inp=10, out=16, kernel_size=3, bias=True):
        super(TestUpsampleNearest2d, self).__init__()
        self.conv2d = nn.Conv2d(inp, out, kernel_size=kernel_size, bias=bias)
        self.up = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x):
        x = self.conv2d(x)
        x = F.upsample(x, scale_factor=2)
        x = self.up(x)
        return x


def get_inputs():
    return [torch.rand([4, 10, 64, 64])]


def get_init_inputs():
    return [[], {}]
