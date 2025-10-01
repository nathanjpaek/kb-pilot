import torch
import torch.optim
import torch.nn as nn


class ColumnMaxPooling(nn.Module):
    """
    take a batch (bs, n_vertices, n_vertices, in_features)
    and returns (bs, n_vertices, in_features)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.max(x, 2)[0]


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
