import torch
from torch import nn
import torch.nn.functional as F


class _DQN(nn.Module):

    def __init__(self, observation_space, action_space):
        super(_DQN, self).__init__()
        self.fc1 = nn.Linear(observation_space, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(4, action_space)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.1)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.1)
        return self.fc3(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'observation_space': 4, 'action_space': 4}]
