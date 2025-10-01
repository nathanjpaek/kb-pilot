import torch
import torch.nn as nn
import torch as th
import torch.utils.data
from collections import *
import torch.nn.init as INIT
from torch.nn import LayerNorm


class HaltingUnit(nn.Module):
    halting_bias_init = 1.0

    def __init__(self, dim_model):
        super(HaltingUnit, self).__init__()
        self.linear = nn.Linear(dim_model, 1)
        self.norm = LayerNorm(dim_model)
        INIT.constant_(self.linear.bias, self.halting_bias_init)

    def forward(self, x):
        return th.sigmoid(self.linear(self.norm(x)))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_model': 4}]
