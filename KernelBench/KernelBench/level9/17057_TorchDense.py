import torch
import numpy as np
import torch.nn as nn


class TorchDense(nn.Module):

    def __init__(self, state_shape, action_size: 'int'):
        super(TorchDense, self).__init__()
        input_size_flatten = self.num_flat_features(state_shape)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.h1 = nn.Linear(input_size_flatten, 256)
        self.h2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, action_size)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.tanh(self.h1(x))
        x = torch.tanh(self.h2(x))
        x = self.out(x)
        return x

    def num_flat_features(self, x):
        return np.prod(x)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'state_shape': 4, 'action_size': 4}]
