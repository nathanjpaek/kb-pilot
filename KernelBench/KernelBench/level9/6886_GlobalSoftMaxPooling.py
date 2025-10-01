import torch
import torch.nn as nn


class GlobalSoftMaxPooling(nn.Module):

    def __init__(self, dim=0):
        super(self.__class__, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.sum(x * torch.softmax(x, dim=self.dim), dim=self.dim)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
