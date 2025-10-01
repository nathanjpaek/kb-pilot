import torch
from torch import nn


class Downscale2d(nn.Module):

    def __init__(self, factor=2):
        super().__init__()
        self.downsample = nn.AvgPool2d(kernel_size=factor, stride=factor)

    def forward(self, x):
        return self.downsample(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
