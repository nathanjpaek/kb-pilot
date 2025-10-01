import torch
from torch import nn
import torch.utils.data


class GatedConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, stride,
        padding, dilation=1, activation=None):
        super(GatedConv2d, self).__init__()
        self.activation = activation
        self.sigmoid = nn.Sigmoid()
        self.h = nn.Conv2d(input_channels, output_channels, kernel_size,
            stride, padding, dilation)
        self.g = nn.Conv2d(input_channels, output_channels, kernel_size,
            stride, padding, dilation)

    def forward(self, x):
        if self.activation is None:
            h = self.h(x)
        else:
            h = self.activation(self.h(x))
        g = self.sigmoid(self.g(x))
        return h * g


class GatedResUnit(nn.Module):

    def __init__(self, input_channels, activation=None):
        super(GatedResUnit, self).__init__()
        self.activation = activation
        self.conv1 = GatedConv2d(input_channels, input_channels, 3, 1, 1, 1,
            activation=activation)
        self.conv2 = GatedConv2d(input_channels, input_channels, 3, 1, 1, 1,
            activation=activation)

    def forward(self, x):
        h1 = self.conv1(x)
        h2 = self.conv2(h1)
        return h2 + x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_channels': 4}]
