import torch
import torch.nn as nn
import torch.utils.data


class Conv(nn.Module):

    def __init__(self, filters0, filters1, kernel_size, bn, bias=True):
        super().__init__()
        if bn:
            bias = False
        self.conv = nn.Conv2d(filters0, filters1, kernel_size, stride=1,
            padding=kernel_size // 2, bias=bias)
        self.bn = nn.BatchNorm2d(filters1) if bn else None

    def forward(self, x):
        h = self.conv(x)
        if self.bn is not None:
            h = self.bn(h)
        return h


class Encoder(nn.Module):

    def __init__(self, input_size, filters):
        super().__init__()
        self.input_size = input_size
        self.conv = Conv(input_size[0], filters, 3, bn=False)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.activation(self.conv(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': [4, 4], 'filters': 4}]
