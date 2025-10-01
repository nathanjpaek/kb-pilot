import torch
import torch.nn as nn
import torch.utils.cpp_extension


class GlobalSumPool2d(nn.Module):

    def forward(self, x):
        return torch.sum(x, [2, 3])


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
