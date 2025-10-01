import torch
from typing import *
from torch import nn


class AddPositionEmbs(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, num_patches: 'int', dim: 'int', dropout_rate:
        'float'=0.0):
        super(AddPositionEmbs, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(dropout_rate, inplace=True
            ) if dropout_rate > 0 else None

    def forward(self, x):
        x = x + self.pos_embedding
        return self.dropout(x) if self.dropout else x


def get_inputs():
    return [torch.rand([4, 4, 5, 4])]


def get_init_inputs():
    return [[], {'num_patches': 4, 'dim': 4}]
