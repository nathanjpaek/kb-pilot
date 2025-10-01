import torch
import torch.nn as nn


class CrossPooling(nn.Module):
    """ Cross pooling """

    def forward(self, x):
        """ Forward function of CrossPooling module.

    Args:
      x: a stack of (batch x channel x height x width) tensors on the last axis.

    Returns:
      A (batch x channel x height x width) tensor after applying max-pooling
        over the last axis.
    """
        x, _ = torch.max(x, dim=-1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
