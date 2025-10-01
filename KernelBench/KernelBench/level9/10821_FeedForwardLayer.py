import torch
from torch import nn


class FeedForwardLayer(nn.Module):

    def __init__(self, hidden_size):
        super(FeedForwardLayer, self).__init__()
        self.linear_1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.linear_2 = nn.Linear(4 * hidden_size, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, hidden_states):
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.relu(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
