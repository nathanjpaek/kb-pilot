import math
import torch
import torch.nn as nn


class PositionalEncoding1d(nn.Module):
    """
    Learning positional embeddings.

    Args:
        shape: Iterable, the shape of the input.
        embedding_dim: int, the size of each embedding vector.
    """

    def __init__(self, size, embedding_dim):
        super(PositionalEncoding1d, self).__init__()
        self.size = size
        self.embedding_dim = embedding_dim
        self.encode_l = nn.Parameter(torch.Tensor(size, 1, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.encode_l, std=0.125 / math.sqrt(self.
            embedding_dim))

    def forward(self, x):
        return x + self.encode_l


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'size': 4, 'embedding_dim': 4}]
