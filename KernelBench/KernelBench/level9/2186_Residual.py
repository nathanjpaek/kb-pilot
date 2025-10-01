import torch
import torch.nn as nn


class Residual(nn.Sequential):
    """ Residual block that runs like a Sequential, but then adds the original input to the output tensor.
        See :class:`torch.nn.Sequential` for more information.

        Warning:
            The dimension between the input and output of the module need to be the same
            or need to be broadcastable from one to the other!
    """

    def forward(self, x):
        y = super().forward(x)
        return x + y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
