import torch
import torch.nn as nn


class GenericLayer(nn.Module):

    def __init__(self, layer, out_channels, padding=(0, 0, 0, 0),
        activation=None):
        super(GenericLayer, self).__init__()
        self._act = activation
        self._layer = layer
        self._norm = nn.InstanceNorm2d(out_channels, affine=True)
        self._pad = nn.ReflectionPad2d(padding)

    def forward(self, x):
        x = self._pad(x)
        x = self._layer(x)
        x = self._norm(x)
        if self._act is not None:
            x = self._act(x)
        return x


class ResidualBlock(nn.Module):

    def __init__(self, channels, kernel_size, stride, padding=(0, 0, 0, 0)):
        super(ResidualBlock, self).__init__()
        self._conv_1 = GenericLayer(nn.Conv2d(128, 128, 3, 1), 128, (1, 1, 
            1, 1), nn.ReLU())
        self._conv_2 = GenericLayer(nn.Conv2d(128, 128, 3, 1), 128, (1, 1, 
            1, 1), nn.ReLU())

    def forward(self, x):
        x = self._conv_1(x)
        x = x + self._conv_2(x)
        return x


def get_inputs():
    return [torch.rand([4, 128, 4, 4])]


def get_init_inputs():
    return [[], {'channels': 4, 'kernel_size': 4, 'stride': 1}]
