import torch
from torch import Tensor
from torch import nn


class Hardsigmoid(nn.Module):

    def __init__(self) ->None:
        super().__init__()

    def forward(self, x: 'Tensor') ->Tensor:
        x = (0.2 * x + 0.5).clamp(min=0.0, max=1.0)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
