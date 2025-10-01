import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Double 3x3 conv + relu
    """

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.relu(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 64, 64])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
