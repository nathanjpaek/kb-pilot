import torch
import torch.nn as nn


class SpatialConv3D(nn.Module):
    """
    Apply 3D conv. over an input signal composed of several input planes with distinct spatial and time axes, by performing 3D convolution over the spatiotemporal axes

    rrgs:
        in_channels (int): number of channels in the input tensor
        out_channels (int): number of channels produced by the convolution
        kernel_size (int or tuple): size of the convolution kernel
        stride (int or tuple): stride
        padding (int or tuple): zero-padding
    """

    def __init__(self, in_channels, out_channels, kernel_size=(1, 3, 3),
        stride=(1, 2, 2), padding=(0, 1, 1)):
        super(SpatialConv3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 16, kernel_size, stride, padding)
        self.reLu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(16, out_channels, kernel_size, stride, padding)
        self.reLu2 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.reLu1(x)
        x = self.conv2(x)
        x = self.reLu2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
