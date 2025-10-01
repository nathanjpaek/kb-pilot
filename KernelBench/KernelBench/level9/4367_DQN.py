import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, obs_size, action_size, seed):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(obs_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, sum(action_size))

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'obs_size': 4, 'action_size': [4, 4], 'seed': 4}]
