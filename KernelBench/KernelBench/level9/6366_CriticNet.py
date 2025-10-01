import torch
import torch.nn as nn
import torch.nn.functional as F


class CriticNet(nn.Module):

    def __init__(self, num_state, num_action):
        super(CriticNet, self).__init__()
        self.num_state = num_state
        self.num_action = num_action
        self.fc1 = nn.Linear(num_state, 100)
        self.fc2 = nn.Linear(100, 100)
        self.v_head = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        state_value = self.v_head(x)
        return state_value


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_state': 4, 'num_action': 4}]
