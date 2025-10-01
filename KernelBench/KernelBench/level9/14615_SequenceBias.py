import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.parallel
from torch.nn.parameter import Parameter


class SequenceBias(nn.Module):
    """
    Adds one bias element to the end of the sequence.
    so if the input has a shape ``(L, N, E)``, where
    ``L`` is the sequence length, ``N`` is the batch size, and ``E`` is
    the embedding dimension, the output will have a shape
    ``(L+1, N, E)``.

    Attributes:
        bias (:class:`torch.nn.parameter.Parameter`): the learnable bias of
            the module of shape ``(E)``, where ``E`` is the embedding dimension.

    Example:
        >>> m = SequenceBias(16)
        >>> input = torch.randn(20, 4, 16)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([21, 4, 16])
    """

    def __init__(self, embed_dim: 'int'):
        """
        Args:
            embed_dim: Embedding dimension
        """
        super(SequenceBias, self).__init__()
        self.bias = Parameter(torch.empty(embed_dim))
        self._reset_parameters()

    def _reset_parameters(self):
        """
        assigns Normally distributed random values to bias.
        """
        nn.init.normal_(self.bias)

    def forward(self, x):
        _, bsz, _ = x.shape
        return torch.cat([x, self.bias.repeat(1, bsz, 1)])


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'embed_dim': 4}]
