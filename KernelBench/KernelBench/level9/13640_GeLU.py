from torch.nn import Module
import functools
import math
import torch
import torch.utils.data
import torch.nn as nn
from torchvision.models import *
import torch.nn.init


class GeLU(Module):

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 
            0.044715 * torch.pow(x, 3))))


class PrePostInitMeta(type):
    """A metaclass that calls optional `__pre_init__` and `__post_init__` methods"""

    def __new__(cls, name, bases, dct):
        x = super().__new__(cls, name, bases, dct)
        old_init = x.__init__

        def _pass(self):
            pass

        @functools.wraps(old_init)
        def _init(self, *args, **kwargs):
            self.__pre_init__()
            old_init(self, *args, **kwargs)
            self.__post_init__()
        x.__init__ = _init
        if not hasattr(x, '__pre_init__'):
            x.__pre_init__ = _pass
        if not hasattr(x, '__post_init__'):
            x.__post_init__ = _pass
        return x


class Module(nn.Module, metaclass=PrePostInitMeta):
    """Same as `nn.Module`, but no need for subclasses to call `super().__init__`"""

    def __pre_init__(self):
        super().__init__()

    def __init__(self):
        pass


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
