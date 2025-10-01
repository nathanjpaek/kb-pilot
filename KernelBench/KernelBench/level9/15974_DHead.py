import torch
import torch.nn as nn
from math import *


class DHead(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(256, 1, 4)

    def forward(self, x):
        output = torch.sigmoid(self.conv(x))
        return output


def get_inputs():
    return [torch.rand([4, 256, 64, 64])]


def get_init_inputs():
    return [[], {}]
