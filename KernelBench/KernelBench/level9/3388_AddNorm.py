import torch
import torch.nn as nn


class AddNorm(nn.Module):
    """
    Applies LayerNorm, Dropout and adds to input. Standard AddNorm operations in Transformers
    """

    def __init__(self, input_dim: 'int', dropout: 'float'):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(input_dim)

    def forward(self, X: 'torch.Tensor', Y: 'torch.Tensor') ->torch.Tensor:
        return self.ln(self.dropout(Y) + X)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'dropout': 0.5}]
