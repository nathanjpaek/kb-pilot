import torch
import torch.nn as nn


class Critic(nn.Module):

    def __init__(self, observation_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(observation_size + action_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, action):
        h = torch.relu(self.fc1(torch.cat([x, action], dim=1)))
        h = torch.relu(self.fc2(h))
        return self.fc3(h)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'observation_size': 4, 'action_size': 4}]
