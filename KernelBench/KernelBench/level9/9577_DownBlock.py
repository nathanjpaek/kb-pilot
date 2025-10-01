import torch
import torch.nn as nn


def get_activation(activation: 'str'):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'leaky':
        return nn.LeakyReLU(negative_slope=0.1)
    elif activation == 'elu':
        return nn.ELU()


def conv_layer(dim: 'int'):
    if dim == 3:
        return nn.Conv3d
    elif dim == 2:
        return nn.Conv2d


def get_conv_layer(in_channels: 'int', out_channels: 'int', kernel_size:
    'int'=3, stride: 'int'=1, padding: 'int'=1, bias: 'bool'=True, dim: 'int'=2
    ):
    return conv_layer(dim)(in_channels, out_channels, kernel_size=
        kernel_size, stride=stride, padding=padding, bias=bias)


def maxpool_layer(dim: 'int'):
    if dim == 3:
        return nn.MaxPool3d
    elif dim == 2:
        return nn.MaxPool2d


def get_maxpool_layer(kernel_size: 'int'=2, stride: 'int'=2, padding: 'int'
    =0, dim: 'int'=2):
    return maxpool_layer(dim=dim)(kernel_size=kernel_size, stride=stride,
        padding=padding)


def get_normalization(normalization: 'str', num_channels: 'int', dim: 'int'):
    if normalization == 'batch':
        if dim == 3:
            return nn.BatchNorm3d(num_channels)
        elif dim == 2:
            return nn.BatchNorm2d(num_channels)
    elif normalization == 'instance':
        if dim == 3:
            return nn.InstanceNorm3d(num_channels)
        elif dim == 2:
            return nn.InstanceNorm2d(num_channels)
    elif 'group' in normalization:
        num_groups = int(normalization.partition('group')[-1])
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


class DownBlock(nn.Module):
    """
    A helper Module that performs 2 Convolutions and 1 MaxPool.
    An activation follows each convolution.
    A normalization layer follows each convolution.
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', pooling:
        'bool'=True, activation: 'str'='relu', normalization: 'str'=None,
        dim: 'str'=2, conv_mode: 'str'='same'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.normalization = normalization
        if conv_mode == 'same':
            self.padding = 1
        elif conv_mode == 'valid':
            self.padding = 0
        self.dim = dim
        self.activation = activation
        self.conv1 = get_conv_layer(self.in_channels, self.out_channels,
            kernel_size=3, stride=1, padding=self.padding, bias=True, dim=
            self.dim)
        self.conv2 = get_conv_layer(self.out_channels, self.out_channels,
            kernel_size=3, stride=1, padding=self.padding, bias=True, dim=
            self.dim)
        if self.pooling:
            self.pool = get_maxpool_layer(kernel_size=2, stride=2, padding=
                0, dim=self.dim)
        self.act1 = get_activation(self.activation)
        self.act2 = get_activation(self.activation)
        if self.normalization:
            self.norm1 = get_normalization(normalization=self.normalization,
                num_channels=self.out_channels, dim=self.dim)
            self.norm2 = get_normalization(normalization=self.normalization,
                num_channels=self.out_channels, dim=self.dim)

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(y)
        if self.normalization:
            y = self.norm1(y)
        y = self.conv2(y)
        y = self.act2(y)
        if self.normalization:
            y = self.norm2(y)
        before_pooling = y
        if self.pooling:
            y = self.pool(y)
        return y, before_pooling


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
