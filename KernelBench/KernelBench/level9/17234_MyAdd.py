import torch
from torch import nn
import torch.nn.parallel
import torch.optim
import torch.utils.data


class MyAdd(nn.Module):

    def __init__(self, size):
        super(MyAdd, self).__init__()
        self.weight = nn.Parameter(torch.rand(size))

    def forward(self, x):
        out = x + self.weight
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'size': 4}]
