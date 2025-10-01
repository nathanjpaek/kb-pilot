import torch
import torch.nn as nn
import torch.utils.data
from torch.nn import init


class ZeroConv2d(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 1, padding=0)
        init.uniform_(self.conv.weight, -0.001, 0.001)
        init.uniform_(self.conv.bias, -0.001, 0.001)
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, x):
        out = self.conv(x)
        out = out * torch.exp(self.scale * 3)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channel': 4, 'out_channel': 4}]
