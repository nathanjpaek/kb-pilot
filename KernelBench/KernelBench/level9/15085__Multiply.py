from torch.nn import Module
import abc
import torch
from torch import Tensor
from torch.nn import Linear
from torch.nn import MSELoss
import torch.nn
from torch import rand


class ConverterModule(Module, abc.ABC):
    """Interface class for test modules for converter."""

    @abc.abstractmethod
    def input_fn(self) ->Tensor:
        """Generate a fitting input for the module.

        Returns:
            an input
        """
        return

    def loss_fn(self) ->Module:
        """The loss function.

        Returns:
            loss function
        """
        return MSELoss()


class _Multiply(ConverterModule):

    def __init__(self):
        super().__init__()
        self.batch_size = 2
        self.in_dim = 4
        out_dim = 3
        self.linear = Linear(self.in_dim, out_dim)

    def forward(self, x):
        x = x * 2.5
        x = self.linear(x)
        x = 0.5 * x
        x = x.multiply(3.1415)
        return x

    def input_fn(self) ->Tensor:
        return rand(self.batch_size, self.in_dim)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
