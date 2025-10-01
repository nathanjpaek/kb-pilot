import torch
import torch.nn as nn


class LocalNet(nn.Module):

    def forward(self, x_in):
        """Defines a double convolution

        :param x_in: input convolutional features
        :returns: convolutional features
        :rtype: Tensor

        """
        x = self.lrelu(self.conv1(self.refpad(x_in)))
        x = self.lrelu(self.conv2(self.refpad(x)))
        return x

    def __init__(self, in_channels=16, out_channels=64):
        """Initialisation function

        :param in_channels:  number of input channels
        :param out_channels: number of output channels
        :returns: N/A
        :rtype: N/A

        """
        super(LocalNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 0, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 0, 1)
        self.lrelu = nn.LeakyReLU()
        self.refpad = nn.ReflectionPad2d(1)


def get_inputs():
    return [torch.rand([4, 16, 4, 4])]


def get_init_inputs():
    return [[], {}]
