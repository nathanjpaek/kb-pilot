import torch
import torch.nn as nn


class GlobalMaxPooling(nn.Module):

    def __init__(self, dim=0):
        super(self.__class__, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.max(dim=self.dim)[0]


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
