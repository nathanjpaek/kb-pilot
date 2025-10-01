import torch
import torch.nn as nn


class AR(nn.Module):

    def __init__(self, window: 'int', hidden_size: 'int'):
        super(AR, self).__init__()
        self.linear = nn.Linear(window, hidden_size)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.linear(x)
        x = torch.transpose(x, 1, 2)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'window': 4, 'hidden_size': 4}]
