import torch
import torch.nn as nn


class SqueezeAndExcite(nn.Module):

    def __init__(self, channels, squeeze_channels, se_ratio):
        super(SqueezeAndExcite, self).__init__()
        squeeze_channels = squeeze_channels * se_ratio
        if not squeeze_channels.is_integer():
            raise ValueError('channels must be divisible by 1 / ratio')
        squeeze_channels = int(squeeze_channels)
        self.se_reduce = nn.Conv2d(channels, squeeze_channels, 1, 1, 0,
            bias=True)
        self.non_linear1 = nn.ReLU(inplace=True)
        self.se_expand = nn.Conv2d(squeeze_channels, channels, 1, 1, 0,
            bias=True)
        self.non_linear2 = nn.Sigmoid()

    def forward(self, x):
        y = torch.mean(x, (2, 3), keepdim=True)
        y = self.non_linear1(self.se_reduce(y))
        y = self.non_linear2(self.se_expand(y))
        y = x * y
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channels': 4, 'squeeze_channels': 4, 'se_ratio': 4}]
