import torch
from torch import nn
import torch.nn.functional as F


class GELU(nn.Module):
    """
    GELU activiation layer.

    Applies the Gaussian Error Linear Units function (w/ dummy inplace arg)
    Described in: https://arxiv.org/abs/1606.08415.

    Args:
        inplace(`Bool`):
            whether use inplace version.
    Returns:
            output tensor after activation.
    """

    def __init__(self, inplace: 'bool'=False) ->None:
        super().__init__()
        if inplace is True:
            pass

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return F.gelu(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
