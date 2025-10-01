import torch
import torch.nn as nn
from torch.testing._internal.common_utils import *


class MyNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.elu = nn.ELU()

    def forward(self, x):
        return self.elu(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
