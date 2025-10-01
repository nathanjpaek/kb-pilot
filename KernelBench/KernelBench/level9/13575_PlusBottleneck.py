import torch
from torch import nn
import torch.nn.parallel


class PlusBottleneck(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

    def forward(self, dec, enc):
        return enc + dec


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
