import torch
import torch.nn as nn


class MaxPool(nn.Module):

    def __init__(self, dim=1):
        super(MaxPool, self).__init__()
        self.dim = dim

    def forward(self, input):
        return torch.max(input, self.dim)[0]

    def __repr__(self):
        return self.__class__.__name__ + ' (' + 'dim=' + str(self.dim) + ')'


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
