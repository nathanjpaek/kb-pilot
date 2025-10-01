import torch
import torch.nn as nn


class BaselineModel(nn.Module):
    """The model that computes V(s)"""

    def __init__(self, n_observations, n_hidden):
        super().__init__()
        self.linear = nn.Linear(n_observations, n_hidden)
        self.linear2 = nn.Linear(n_hidden, 1)

    def forward(self, frame):
        z = torch.tanh(self.linear(frame))
        critic = self.linear2(z)
        return critic


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_observations': 4, 'n_hidden': 4}]
