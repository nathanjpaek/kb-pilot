import torch
import torch.nn as nn
import torch.nn.utils as utils
from torchvision import utils


class CustomConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=None, bias=True, spectral_norm=False, residual_init=True):
        super(CustomConv2d, self).__init__()
        self.residual_init = residual_init
        if padding is None:
            padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias)
        if spectral_norm:
            self.conv = utils.spectral_norm(self.conv)

    def forward(self, input):
        return self.conv(input)


class ConvMeanPool(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=True,
        spectral_norm=False, residual_init=True):
        super(ConvMeanPool, self).__init__()
        self.conv = CustomConv2d(in_channels, out_channels, kernel_size,
            bias=bias, spectral_norm=spectral_norm, residual_init=residual_init
            )

    def forward(self, input):
        output = input
        output = self.conv(output)
        output = (output[:, :, ::2, ::2] + output[:, :, 1::2, ::2] + output
            [:, :, ::2, 1::2] + output[:, :, 1::2, 1::2]) / 4
        return output


class MeanPoolConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=True,
        spectral_norm=False, residual_init=True):
        super(MeanPoolConv, self).__init__()
        self.conv = CustomConv2d(in_channels, out_channels, kernel_size,
            bias=bias, spectral_norm=spectral_norm, residual_init=residual_init
            )

    def forward(self, input):
        output = input
        output = (output[:, :, ::2, ::2] + output[:, :, 1::2, ::2] + output
            [:, :, ::2, 1::2] + output[:, :, 1::2, 1::2]) / 4
        output = self.conv(output)
        return output


class OptimizedResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
        spectral_norm=False):
        super(OptimizedResidualBlock, self).__init__()
        self.conv1 = CustomConv2d(in_channels, out_channels, kernel_size=
            kernel_size, spectral_norm=spectral_norm)
        self.conv2 = ConvMeanPool(out_channels, out_channels, kernel_size=
            kernel_size, spectral_norm=spectral_norm)
        self.conv_shortcut = MeanPoolConv(in_channels, out_channels,
            kernel_size=1, spectral_norm=spectral_norm, residual_init=False)
        self.relu2 = nn.ReLU()

    def forward(self, input):
        shortcut = self.conv_shortcut(input)
        output = input
        output = self.conv1(output)
        output = self.relu2(output)
        output = self.conv2(output)
        return shortcut + output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
