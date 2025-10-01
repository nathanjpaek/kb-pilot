import torch
import torch.nn as nn


class GenNoise(nn.Module):

    def __init__(self, dim2):
        super(GenNoise, self).__init__()
        self.dim2 = dim2

    def forward(self, x):
        a = list(x.size())
        a[1] = self.dim2
        b = torch.zeros(a).type_as(x.data)
        b.normal_()
        x = torch.autograd.Variable(b)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim2': 4}]
