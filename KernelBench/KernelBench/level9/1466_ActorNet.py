import torch
import torch.nn as nn
from torch.nn import functional as F


class ActorNet(nn.Module):

    def __init__(self, obs_size, act_size, high_action=1):
        super().__init__()
        self.high_action = high_action
        self.base = nn.Linear(obs_size, 400)
        self.fc1 = nn.Linear(400, 300)
        self.fc2 = nn.Linear(300, 300)
        self.actions = nn.Linear(300, act_size)

    def forward(self, x):
        y = F.relu(self.base(x))
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = torch.tanh(self.actions(y))
        return self.high_action * y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'obs_size': 4, 'act_size': 4}]
