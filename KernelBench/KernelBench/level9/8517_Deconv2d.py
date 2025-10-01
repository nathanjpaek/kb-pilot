import torch
import torch.nn as nn


class Deconv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bn
        =False, activation='leakyrelu', dropout=False):
        super(Deconv2d, self).__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.ConvTranspose2d(in_channels, out_channels,
            kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0,
            affine=True) if bn else None
        self.dropout = nn.Dropout(p=0.5) if dropout else None
        if activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError('Not a valid activation, received {}'.format(
                activation))

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.activation(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
