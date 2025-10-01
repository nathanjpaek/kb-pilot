import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from itertools import product as product


class Scale(nn.Module):

    def __init__(self, channels):
        super(Scale, self).__init__()
        self.weight = Parameter(torch.Tensor(channels))
        self.bias = Parameter(torch.Tensor(channels))
        self.channels = channels

    def forward(self, x):
        nB = x.size(0)
        nC = x.size(1)
        nH = x.size(2)
        nW = x.size(3)
        x = x * self.weight.view(1, nC, 1, 1).expand(nB, nC, nH, nW
            ) + self.bias.view(1, nC, 1, 1).expand(nB, nC, nH, nW)
        return x

    def __repr__(self):
        return 'Scale(channels=%d)' % self.channels


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channels': 4}]
