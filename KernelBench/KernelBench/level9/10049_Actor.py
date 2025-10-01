import torch
import torch.nn.functional as F
from torch import nn


class Actor(nn.Module):
    """
    Policy Network (state --> action)
    """

    def __init__(self, state_size: 'int', action_size: 'int', hidden_size:
        'int'=256):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        act = torch.tanh(self.out(x))
        return act


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_size': 4, 'action_size': 4}]
