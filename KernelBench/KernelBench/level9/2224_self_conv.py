import torch
import torch.nn as nn
import torch.nn.functional as F


def quantize_w(x):
    x = Q_W.apply(x)
    return x


def fw(x, bitW):
    if bitW == 32:
        return x
    x = quantize_w(x)
    return x


class Q_W(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        return x.sign() * x.abs().mean()

    @staticmethod
    def backward(ctx, grad):
        return grad


class self_conv(nn.Conv2d):

    def __init__(self, in_channels, out_channels, bitW, kernel_size, stride
        =1, padding=0, bias=False):
        super(self_conv, self).__init__(in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, bias=bias)
        self.bitW = bitW
        self.padding = padding
        self.stride = stride

    def forward(self, input):
        if self.padding > 0:
            padding_shape = (self.padding, self.padding, self.padding, self
                .padding)
            input = F.pad(input, padding_shape, 'constant', 1)
        output = F.conv2d(input, fw(self.weight, self.bitW), bias=self.bias,
            stride=self.stride, dilation=self.dilation, groups=self.groups)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'bitW': 4,
        'kernel_size': 4}]
