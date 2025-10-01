import torch
import torch.nn as nn
from random import *


class C1Bilinear(nn.Module):

    def __init__(self, num_class=150, fc_dim=4096, segSize=384, use_softmax
        =False):
        super(C1Bilinear, self).__init__()
        self.segSize = segSize
        self.use_softmax = use_softmax
        self.conv_last = nn.Conv2d(fc_dim, num_class, 1, 1, 0, bias=False)

    def forward(self, x, segSize=None):
        if segSize is None:
            segSize = self.segSize, self.segSize
        elif isinstance(segSize, int):
            segSize = segSize, segSize
        x = self.conv_last(x)
        if not (x.size(2) == segSize[0] and x.size(3) == segSize[1]):
            x = nn.functional.upsample(x, size=segSize, mode='bilinear')
        if self.use_softmax:
            x = nn.functional.softmax(x)
        else:
            x = nn.functional.log_softmax(x)
        return x


def get_inputs():
    return [torch.rand([4, 4096, 64, 64])]


def get_init_inputs():
    return [[], {}]
