import torch
import torch.nn as nn


class ConcatBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConcatBlock, self).__init__()
        self.in_chns = in_channels
        self.out_chns = out_channels
        self.conv1 = nn.Conv2d(self.in_chns, self.in_chns, kernel_size=1,
            padding=0)
        self.conv2 = nn.Conv2d(self.in_chns, self.out_chns, kernel_size=1,
            padding=0)
        self.ac1 = nn.LeakyReLU()
        self.ac2 = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.ac1(x)
        x = self.conv2(x)
        x = self.ac2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
