import torch
from typing import *


class SentinelMBSI(torch.nn.Module):

    def __init__(self, band_count):
        super(SentinelMBSI, self).__init__()
        self.no_weights = True

    def forward(self, x):
        self.red = x[:, 3:4, :, :]
        self.green = x[:, 2:3, :, :]
        return 2 * (self.red - self.green) / (self.red + self.green - 2 * (
            1 << 16))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'band_count': 4}]
