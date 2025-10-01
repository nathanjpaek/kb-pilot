import torch
import torch.nn as nn


class SpatialTemporalConv3D(nn.Module):
    """
    Apply 3D conv. over an input signal composed of several input planes with distinct spatial and time axes, by performing 3D convolution over the spatiotemporal axes

    args:
        in_channels (int): number of channels in the input tensor
        out_channels (int): number of channels produced by the convolution
        kernel_size (int or tuple): size of the convolution kernel
        stride (int or tuple): stride
        padding (int or tuple): zero-padding
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
        padding=1):
        super(SpatialTemporalConv3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size, stride, padding)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(64, 64, kernel_size, stride, padding)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.conv3 = nn.Conv3d(64, 32, kernel_size, stride, padding)
        self.relu3 = nn.LeakyReLU(inplace=True)
        self.conv4 = nn.Conv3d(32, out_channels, kernel_size, stride, padding)
        self.relu4 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
