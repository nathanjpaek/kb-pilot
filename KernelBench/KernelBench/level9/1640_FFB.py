import torch
import torch.nn as nn
from math import sqrt as sqrt
from itertools import product as product


class FFB(nn.Module):

    def __init__(self, c1, c2, size):
        super(FFB, self).__init__()
        self.conv_y1 = nn.Conv2d(c1, c1, kernel_size=1)
        self.act1 = nn.ReLU(inplace=True)
        self.conv_y2 = nn.Conv2d(c2, c1, kernel_size=1)
        self.act2 = nn.ReLU(inplace=True)
        self.up = nn.Upsample(size=size, mode='nearest')

    def forward(self, x, y):
        y1 = self.conv_y1(x)
        y2 = self.conv_y2(y)
        y2 = self.up(y2)
        return y1 + y2


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'c1': 4, 'c2': 4, 'size': 4}]
