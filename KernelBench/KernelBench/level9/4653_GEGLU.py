import torch
from torch import Tensor
import torch.nn.functional as f
from torch import nn


class GEGLU(nn.Module):
    """Gated GELU, it splits a tensor in two slices based on the last dimension, and then multiply the
       first half and the gelu of the second half
    """

    def forward(self, x: 'Tensor') ->Tensor:
        x, gates = x.chunk(2, dim=-1)
        return x * f.gelu(gates)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
