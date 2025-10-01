import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F


def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True, groups=1
    ):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=
        stride, padding=padding, bias=bias, groups=groups)


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels, pooling=True,
        dropout_rate=0.5):
        super(DownConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.dropout_rate = dropout_rate
        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate)
        return x, before_pool


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
