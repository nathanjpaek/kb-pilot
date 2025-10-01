import torch
from torch import nn


class SE(nn.Module):

    def __init__(self, channels, se_ratio):
        super(SE, self).__init__()
        inter_channels = max(1, int(channels * se_ratio))
        self.conv1 = nn.Conv2d(channels, inter_channels, (1, 1))
        self.silu = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(inter_channels, channels, (1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.silu(self.conv1(input))
        x = self.sigmoid(self.conv2(x))
        return x * input


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channels': 4, 'se_ratio': 4}]
