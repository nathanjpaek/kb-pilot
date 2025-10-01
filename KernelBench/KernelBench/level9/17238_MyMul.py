import torch
from torch import nn
import torch.nn.parallel
import torch.optim
import torch.utils.data


class MyMul(nn.Module):

    def __init__(self, size):
        super(MyMul, self).__init__()
        self.weight = nn.Parameter(torch.rand(1))

    def forward(self, x):
        out = x * torch.abs(self.weight)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'size': 4}]
