import torch
from torch import nn


class ConvRelu(nn.Module):

    def __init__(self, in_: 'int', out: 'int', activate=True):
        super(ConvRelu, self).__init__()
        self.activate = activate
        self.conv = nn.Conv2d(in_, out, 3, padding=1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.activate:
            x = self.activation(x)
        return x


class ResidualBlock(nn.Module):

    def __init__(self, in_channels: 'int', num_filters: 'int',
        batch_activate=False):
        super(ResidualBlock, self).__init__()
        self.batch_activate = batch_activate
        self.activation = nn.ReLU(inplace=True)
        self.conv_block = ConvRelu(in_channels, num_filters, activate=True)
        self.conv_block_na = ConvRelu(in_channels, num_filters, activate=False)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inp):
        x = self.conv_block(inp)
        x = self.conv_block_na(x)
        x = x.add(inp)
        if self.batch_activate:
            x = self.activation(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'num_filters': 4}]
