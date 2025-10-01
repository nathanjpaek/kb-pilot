import torch
from torch import nn


class ConvBlock(torch.nn.Module):

    def __init__(self, input_size, output_size, kernel_size, stride,
        padding, bias=True):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size,
            stride, padding, bias=bias)
        self.act = torch.nn.PReLU()
        self.bn = nn.InstanceNorm2d(output_size)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return self.act(out)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'output_size': 4, 'kernel_size': 4,
        'stride': 1, 'padding': 4}]
