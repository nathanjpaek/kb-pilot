import torch
import torch.nn as nn
import torch.nn.parallel


class Sigmoid(nn.Module):

    def __init__(self, inplace: 'bool'=False):
        super(Sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.sigmoid_() if self.inplace else x.sigmoid()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
