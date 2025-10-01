import torch
from torch import Tensor
import torch.nn as nn
from torch import tanh


class ScaledTanh(nn.Module):

    def __init__(self, factor):
        super(ScaledTanh, self).__init__()
        self.factor = factor

    def forward(self, inputs: 'Tensor') ->Tensor:
        return tanh(inputs) * self.factor


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'factor': 4}]
