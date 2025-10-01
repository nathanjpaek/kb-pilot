import torch
import torch.nn as nn
from math import sqrt as sqrt
from itertools import product as product


class LDS(nn.Module):

    def __init__(self):
        super(LDS, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)

    def forward(self, x):
        x_pool1 = self.pool1(x)
        x_pool2 = self.pool2(x_pool1)
        x_pool3 = self.pool3(x_pool2)
        return x_pool3


def get_inputs():
    return [torch.rand([4, 4, 64, 64])]


def get_init_inputs():
    return [[], {}]
