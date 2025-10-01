import torch
import torch.nn as nn


class TestConvTranspose2d(nn.Module):
    """Module for Dense conversion testing
    """

    def __init__(self, inp=10, out=16, kernel_size=3, bias=True):
        super(TestConvTranspose2d, self).__init__()
        self.conv2d = nn.ConvTranspose2d(inp, out, padding=1, stride=2,
            kernel_size=kernel_size, bias=bias)

    def forward(self, x):
        x = self.conv2d(x)
        return x


def get_inputs():
    return [torch.rand([4, 10, 4, 4])]


def get_init_inputs():
    return [[], {}]
