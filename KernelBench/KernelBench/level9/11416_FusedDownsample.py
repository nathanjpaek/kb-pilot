import torch
from torch import nn
from torch.nn import functional as F
from math import sqrt


class FusedDownsample(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()
        weight = torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        bias = torch.zeros(out_channel)
        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)
        self.pad = padding

    def forward(self, input):
        weight = F.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (weight[:, :, 1:, 1:] + weight[:, :, :-1, 1:] + weight[:,
            :, 1:, :-1] + weight[:, :, :-1, :-1]) / 4
        out = F.conv2d(input, weight, self.bias, stride=2, padding=self.pad)
        return out


def get_inputs():
    return [torch.rand([4, 4, 64, 64])]


def get_init_inputs():
    return [[], {'in_channel': 4, 'out_channel': 4, 'kernel_size': 4}]
