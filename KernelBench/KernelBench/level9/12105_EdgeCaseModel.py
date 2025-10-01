import torch
from typing import Any
import torch.nn as nn


class LayerWithRidiculouslyLongNameAndDoesntDoAnything(nn.Module):
    """ Model with a very long name. """

    def __init__(self) ->None:
        super().__init__()
        self.identity = nn.Identity()

    def forward(self, x: 'Any') ->Any:
        return self.identity(x)


class EdgeCaseModel(nn.Module):
    """ Model that throws an exception when used. """

    def __init__(self, throw_error: 'bool'=False, return_str: 'bool'=False,
        return_class: 'bool'=False) ->None:
        super().__init__()
        self.throw_error = throw_error
        self.return_str = return_str
        self.return_class = return_class
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.model = LayerWithRidiculouslyLongNameAndDoesntDoAnything()

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self.conv1(x)
        x = self.model('string output' if self.return_str else x)
        if self.throw_error:
            x = self.conv1(x)
        if self.return_class:
            x = self.model(EdgeCaseModel)
        return x


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
