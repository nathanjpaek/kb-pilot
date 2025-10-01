import torch
import torch.nn as nn
from math import sqrt as sqrt
from itertools import product as product


class LocAndConf(nn.Module):

    def __init__(self, c_in, c_out, num_classes):
        super(LocAndConf, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.num_classes = num_classes
        self.conv_loc = nn.Conv2d(c_in, c_out * 4, kernel_size=3, padding=1)
        self.conv_conf = nn.Conv2d(c_in, c_out * num_classes, kernel_size=3,
            padding=1)

    def forward(self, x):
        loc = self.conv_loc(x)
        conf = self.conv_conf(x)
        return loc, conf


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'c_in': 4, 'c_out': 4, 'num_classes': 4}]
