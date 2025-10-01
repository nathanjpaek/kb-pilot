import torch
import torch.nn as nn


class PositionalAttention(nn.Module):
    """
        A simple positional attention layer that assigns different weights for word in different relative position.
    """

    def __init__(self, n_seq):
        super(PositionalAttention, self).__init__()
        self.pos_att = nn.Parameter(torch.ones(n_seq))

    def forward(self, x):
        return (x.transpose(-2, -1) * self.pos_att).transpose(-2, -1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_seq': 4}]
