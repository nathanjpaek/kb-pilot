import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


def hidden_unit(layer):
    inp = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(inp)
    return -lim, lim


class Actor(nn.Module):

    def __init__(self, state_size, action_size, seed=2, fc_units=256):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc_units)
        self.fc2 = nn.Linear(fc_units, action_size)
        self.reset_weights()

    def reset_weights(self):
        self.fc1.weight.data.uniform_(*hidden_unit(self.fc1))
        self.fc2.weight.data.uniform_(-0.003, 0.003)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        return torch.tanh(self.fc2(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_size': 4, 'action_size': 4}]
