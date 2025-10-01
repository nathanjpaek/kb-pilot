import torch
from torch import nn


class IWConv2d(nn.Module):

    def __init__(self, input_dim, output_dim, kernel_size, he_init=True,
        stride=1, bias=True):
        super(IWConv2d, self).__init__()
        self.he_init = he_init
        self.padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1,
            padding=self.padding, bias=bias)

    def forward(self, input):
        output = self.conv(input)
        return output


class ConvMeanPool(nn.Module):

    def __init__(self, input_dim, output_dim, kernel_size, he_init=True):
        super(ConvMeanPool, self).__init__()
        self.he_init = he_init
        self.conv = IWConv2d(input_dim, output_dim, kernel_size, he_init=
            self.he_init)

    def forward(self, input):
        output = self.conv(input)
        output = (output[:, :, ::2, ::2] + output[:, :, 1::2, ::2] + output
            [:, :, ::2, 1::2] + output[:, :, 1::2, 1::2]) / 4
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4, 'kernel_size': 4}]
