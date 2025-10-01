import torch
from torch import nn


class MyLinear(nn.Module):

    def __init__(self, inp, outp):
        super(MyLinear, self).__init__()
        self.w = nn.Parameter(torch.randn(outp, inp))
        self.b = nn.Parameter(torch.randn(outp))

    def forward(self, x):
        x = x @ self.w.t() + self.b
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inp': 4, 'outp': 4}]
