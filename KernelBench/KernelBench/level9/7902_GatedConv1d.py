import torch
import torch.nn as nn
import torch.nn


class GatedConv1d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, stride,
        padding=0, dilation=1, activation=None):
        super(GatedConv1d, self).__init__()
        self.activation = activation
        self.sigmoid = nn.Sigmoid()
        self.h = nn.Conv1d(input_channels, output_channels, kernel_size,
            stride, padding, dilation)
        self.g = nn.Conv1d(input_channels, output_channels, kernel_size,
            stride, padding, dilation)

    def forward(self, x):
        if self.activation is None:
            h = self.h(x)
        else:
            h = self.activation(self.h(x))
        g = self.sigmoid(self.g(x))
        return h * g


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_channels': 4, 'output_channels': 4, 'kernel_size': 
        4, 'stride': 1}]
