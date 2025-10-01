import torch
import torch as th
from torch import nn as nn


class EnsembleLayer(nn.Module):

    def __init__(self, ensemble_size, input_dim, output_dim):
        super().__init__()
        self.W = nn.Parameter(th.empty((ensemble_size, input_dim,
            output_dim)), requires_grad=True).float()
        nn.init.xavier_uniform_(self.W, gain=nn.init.calculate_gain('relu'))
        self.b = nn.Parameter(th.zeros((ensemble_size, 1, output_dim)),
            requires_grad=True).float()

    def forward(self, x):
        return x @ self.W + self.b


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'ensemble_size': 4, 'input_dim': 4, 'output_dim': 4}]
