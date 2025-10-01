from torch.nn import Module
import functools
import torch
import torch.nn as nn
from typing import *


def sigmoid_range(x, low, high):
    """Sigmoid function with range `(low, high)`"""
    return torch.sigmoid(x) * (high - low) + low


class PrePostInitMeta(type):
    """A metaclass that calls optional `__pre_init__` and `__post_init__` methods"""

    def __new__(cls, name, bases, dct):
        x = super().__new__(cls, name, bases, dct)

        def _pass(self, *args, **kwargs):
            pass
        for o in ('__init__', '__pre_init__', '__post_init__'):
            if not hasattr(x, o):
                setattr(x, o, _pass)
        old_init = x.__init__

        @functools.wraps(old_init)
        def _init(self, *args, **kwargs):
            self.__pre_init__()
            old_init(self, *args, **kwargs)
            self.__post_init__()
        setattr(x, '__init__', _init)
        return x


class Module(nn.Module, metaclass=PrePostInitMeta):
    """Same as `nn.Module`, but no need for subclasses to call `super().__init__`"""

    def __pre_init__(self):
        super().__init__()

    def __init__(self):
        pass


class SigmoidRange(Module):
    """Sigmoid module with range `(low, high)`"""

    def __init__(self, low, high):
        self.low, self.high = low, high

    def forward(self, x):
        return sigmoid_range(x, self.low, self.high)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'low': 4, 'high': 4}]
