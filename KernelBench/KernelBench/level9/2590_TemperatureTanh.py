import torch
from torch import Tensor
from torch.functional import Tensor
from torch import nn as nn


class TemperatureTanh(nn.Module):

    def __init__(self, temperature: 'float'=1.0) ->None:
        """The hyperbolic tangent with an optional temperature."""
        super().__init__()
        assert temperature != 0.0, 'temperature must be nonzero.'
        self._T = temperature
        self.tanh = torch.nn.Tanh()

    def forward(self, x: 'Tensor') ->Tensor:
        return self.tanh(x / self._T)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
