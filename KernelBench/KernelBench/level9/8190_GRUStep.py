import torch
from torch import nn
import torch.nn.modules.loss
from scipy.sparse import *


class GRUStep(nn.Module):

    def __init__(self, hidden_size, input_size):
        super(GRUStep, self).__init__()
        """GRU module"""
        self.linear_z = nn.Linear(hidden_size + input_size, hidden_size,
            bias=False)
        self.linear_r = nn.Linear(hidden_size + input_size, hidden_size,
            bias=False)
        self.linear_t = nn.Linear(hidden_size + input_size, hidden_size,
            bias=False)

    def forward(self, h_state, input):
        z = torch.sigmoid(self.linear_z(torch.cat([h_state, input], -1)))
        r = torch.sigmoid(self.linear_r(torch.cat([h_state, input], -1)))
        t = torch.tanh(self.linear_t(torch.cat([r * h_state, input], -1)))
        h_state = (1 - z) * h_state + z * t
        return h_state


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4, 'input_size': 4}]
