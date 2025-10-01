import torch
from torch import Tensor
import torch.nn as nn


class FCLayer(nn.Module):

    def __init__(self, input_dim: 'int', output_dim: 'int', dropout_rate:
        'float'=0.0, use_activation: 'bool'=True) ->None:
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x: 'Tensor') ->Tensor:
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4}]
