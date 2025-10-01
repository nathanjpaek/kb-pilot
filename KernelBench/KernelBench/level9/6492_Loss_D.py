import torch
import torch.nn as nn
from numpy import *


class Loss_D(nn.Module):
    """docstring for Loss_D"""

    def __init__(self):
        super(Loss_D, self).__init__()

    def forward(self, input_h):
        return -input_h * torch.log(input_h)
        pass


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
