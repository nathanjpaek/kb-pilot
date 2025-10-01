import torch
import numpy as np
import torch.nn as nn


class ValueFunction(nn.Module):

    def __init__(self, width, n_states):
        super(ValueFunction, self).__init__()
        self.linear1 = nn.Linear(n_states, width)
        nn.init.normal_(self.linear1.weight, 0.0, 1 / np.sqrt(n_states))
        torch.nn.init.constant_(self.linear1.bias, 0.0)
        self.linear2 = nn.Linear(width, 1)
        nn.init.normal_(self.linear2.weight, 0.0, 1 / np.sqrt(width))
        torch.nn.init.constant_(self.linear2.bias, 0.0)

    def forward(self, x):
        x = torch.tanh(self.linear1(x))
        value = self.linear2(x)
        return value


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'width': 4, 'n_states': 4}]
