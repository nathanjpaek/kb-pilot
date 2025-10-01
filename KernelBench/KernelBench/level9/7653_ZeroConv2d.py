import torch
import torch.nn as nn


class ZeroConv2d(nn.Module):

    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()
        self.in_channel = in_channel
        self.conv = nn.Conv2d(in_channel, out_channel, [1, 1], padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input):
        out = self.conv(input)
        out = out * torch.exp(self.scale * 3)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channel': 4, 'out_channel': 4}]
