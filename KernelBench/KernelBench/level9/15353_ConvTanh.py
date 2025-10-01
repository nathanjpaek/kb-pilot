import torch
import numpy as np


class ConvLayer(torch.nn.Module):
    """Reflection padded convolution layer."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, bias
        =True):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels,
            kernel_size, stride=stride, bias=bias)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ConvTanh(ConvLayer):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvTanh, self).__init__(in_channels, out_channels,
            kernel_size, stride)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        out = super(ConvTanh, self).forward(x)
        return self.tanh(out / 255) * 150 + 255 / 2


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4,
        'stride': 1}]
