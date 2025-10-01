import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.parallel
from torch.nn.parameter import Parameter


class SequenceBias(nn.Module):
    """ Adds one bias element to the end of the sequence
    Args:
        embed_dim: Embedding dimension

    Shape:
        - Input: (L, N, E), where
            L - sequence length, N - batch size, E - embedding dimension
        - Output: (L+1, N, E), where
            L - sequence length, N - batch size, E - embedding dimension

    Attributes:
        bias:   the learnable bias of the module of shape (E),
            where E - embedding dimension

    Examples::

        >>> m = SequenceBias(16)
        >>> input = torch.randn(20, 4, 16)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([21, 4, 16])
    """

    def __init__(self, embed_dim):
        super(SequenceBias, self).__init__()
        self.bias = Parameter(torch.empty(embed_dim))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.bias)

    def forward(self, x):
        _, bsz, _ = x.shape
        return torch.cat([x, self.bias.repeat(1, bsz, 1)])


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'embed_dim': 4}]
