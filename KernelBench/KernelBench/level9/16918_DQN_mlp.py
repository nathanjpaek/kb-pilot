import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data


class DQN_mlp(nn.Module):
    """Layers for a Deep Q Network, based on a simple MLP."""

    def __init__(self, m, n, num_actions, num_hidden1=1000, num_hidden2=2000):
        super(DQN_mlp, self).__init__()
        self.m = m
        self.n = n
        self.num_hidden1 = num_hidden1
        self.num_hidden2 = num_hidden2
        self.fc1 = nn.Linear(m * n, num_hidden1)
        self.fc2 = nn.Linear(num_hidden1, num_hidden2)
        self.fc3 = nn.Linear(num_hidden2, num_hidden2)
        self.fc4 = nn.Linear(num_hidden2, num_actions)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'m': 4, 'n': 4, 'num_actions': 4}]
