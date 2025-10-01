import torch
import torch.nn.functional as F
from torch import nn


class Critic(nn.Module):
    """
    Value Network (state + action --> value)
    """

    def __init__(self, state_size: 'int', action_size: 'int', hidden_size:
        'int'=256):
        super().__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        val = self.out(x)
        return val


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'state_size': 4, 'action_size': 4}]
