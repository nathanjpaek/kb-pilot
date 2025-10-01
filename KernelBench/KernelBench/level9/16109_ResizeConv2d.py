import torch
import torch.utils.data
import torch.nn as nn


class ResizeConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, stride,
        padding, dilation=1, scale_factor=2, activation=None):
        super(ResizeConv2d, self).__init__()
        self.activation = activation
        self.upsamplingNN = nn.Upsample(scale_factor=scale_factor)
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size,
            stride, padding, dilation)

    def forward(self, x):
        h = self.upsamplingNN(x)
        h = self.conv(h)
        if self.activation is None:
            out = h
        else:
            out = self.activation(h)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_channels': 4, 'output_channels': 4, 'kernel_size': 
        4, 'stride': 1, 'padding': 4}]
