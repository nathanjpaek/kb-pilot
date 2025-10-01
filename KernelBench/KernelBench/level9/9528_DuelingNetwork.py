import torch
import torch.nn.functional as F
import torch.nn as nn


class DuelingNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed):
        super(DuelingNetwork, self).__init__()
        torch.manual_seed(seed)
        hidden1 = 64
        hidden2 = 64
        self.fc1 = nn.Linear(state_size, hidden1)
        self.vfc1 = nn.Linear(hidden1, hidden2)
        self.vfc2 = nn.Linear(hidden2, 1)
        self.afc1 = nn.Linear(hidden1, hidden2)
        self.afc2 = nn.Linear(hidden2, action_size)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        v = self.vfc1(x)
        v = F.relu(v)
        v = self.vfc2(v)
        a = self.afc1(x)
        a = F.relu(a)
        a = self.afc2(a)
        q = v + a - a.mean()
        return q


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_size': 4, 'action_size': 4, 'seed': 4}]
