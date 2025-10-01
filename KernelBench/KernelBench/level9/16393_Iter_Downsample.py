import torch
import torch.nn as nn
from math import sqrt as sqrt
from itertools import product as product


class Iter_Downsample(nn.Module):

    def __init__(self):
        super(Iter_Downsample, self).__init__()
        self.init_ds = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2,
            padding=0), nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.ds1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.ds2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.ds3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.ds4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.init_ds(x)
        x1 = self.ds1(x)
        x2 = self.ds2(x1)
        x3 = self.ds3(x2)
        x4 = self.ds4(x3)
        return x1, x2, x3, x4


def get_inputs():
    return [torch.rand([4, 4, 64, 64])]


def get_init_inputs():
    return [[], {}]
