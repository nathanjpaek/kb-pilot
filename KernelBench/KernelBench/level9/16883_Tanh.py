import torch
import torch.nn as nn
import torch.utils.data


class Tanh(nn.Module):

    def __init__(self, inplace: 'bool'=False):
        super(Tanh, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.tanh_() if self.inplace else x.tanh()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
