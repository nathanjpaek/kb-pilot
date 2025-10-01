import torch
import torch.nn as nn


class GenericLayer(nn.Module):

    def __init__(self, layer, out_channels, padding=(0, 0, 0, 0),
        activation=None):
        super(GenericLayer, self).__init__()
        self._act = activation
        self._layer = layer
        self._norm = nn.InstanceNorm2d(out_channels, affine=True)
        self._pad = nn.ReflectionPad2d(padding)

    def forward(self, x):
        x = self._pad(x)
        x = self._layer(x)
        x = self._norm(x)
        if self._act is not None:
            x = self._act(x)
        return x


class ResidualBlock(nn.Module):

    def __init__(self, channels, kernel_size, stride, padding=(0, 0, 0, 0)):
        super(ResidualBlock, self).__init__()
        self._conv_1 = GenericLayer(nn.Conv2d(128, 128, 3, 1), 128, (1, 1, 
            1, 1), nn.ReLU())
        self._conv_2 = GenericLayer(nn.Conv2d(128, 128, 3, 1), 128, (1, 1, 
            1, 1), nn.ReLU())

    def forward(self, x):
        x = self._conv_1(x)
        x = x + self._conv_2(x)
        return x


class UpsampleConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        scale_factor):
        super(UpsampleConvLayer, self).__init__()
        self._scale_factor = scale_factor
        self._reflection_pad = nn.ReflectionPad2d(kernel_size // 2)
        self._conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x = nn.functional.interpolate(x, mode='nearest', scale_factor=self.
            _scale_factor)
        x = self._reflection_pad(x)
        x = self._conv(x)
        return x


class TransferNet(nn.Module):

    def __init__(self):
        super(TransferNet, self).__init__()
        self._conv_1 = GenericLayer(nn.Conv2d(3, 32, 9, 1), 32, (5, 5, 5, 5
            ), nn.ReLU())
        self._conv_2 = GenericLayer(nn.Conv2d(32, 64, 3, 2), 64, (1, 0, 1, 
            0), nn.ReLU())
        self._conv_3 = GenericLayer(nn.Conv2d(64, 128, 3, 2), 128, (1, 0, 1,
            0), nn.ReLU())
        self._res_1 = ResidualBlock(128, 3, 1, (1, 1, 1, 1))
        self._res_2 = ResidualBlock(128, 3, 1, (1, 1, 1, 1))
        self._res_3 = ResidualBlock(128, 3, 1, (1, 1, 1, 1))
        self._res_4 = ResidualBlock(128, 3, 1, (1, 1, 1, 1))
        self._res_5 = ResidualBlock(128, 3, 1, (1, 1, 1, 1))
        self._conv_4 = GenericLayer(UpsampleConvLayer(128, 64, 3, 1, 2), 64,
            (0, 0, 0, 0), nn.ReLU())
        self._conv_5 = GenericLayer(UpsampleConvLayer(64, 32, 3, 1, 2), 32,
            (0, 0, 0, 0), nn.ReLU())
        self._conv_6 = GenericLayer(nn.Conv2d(32, 3, 9, 1), 3, (4, 4, 4, 4),
            nn.Sigmoid())

    def forward(self, x):
        x = self._conv_1(x)
        x = self._conv_2(x)
        x = self._conv_3(x)
        x = self._res_1(x)
        x = self._res_2(x)
        x = self._res_3(x)
        x = self._res_4(x)
        x = self._res_5(x)
        x = self._conv_4(x)
        x = self._conv_5(x)
        x = self._conv_6(x)
        return x


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
