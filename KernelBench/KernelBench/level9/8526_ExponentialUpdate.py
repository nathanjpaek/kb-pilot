import torch
from torch import Tensor
from torch import nn
from torch.jit import Final


class ExponentialUpdate(nn.Module):
    alpha: 'Final[int]'

    def __init__(self, alpha: 'float'):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x: 'Tensor', state: 'Tensor') ->Tensor:
        return x * (1 - self.alpha) + state * self.alpha


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'alpha': 4}]
