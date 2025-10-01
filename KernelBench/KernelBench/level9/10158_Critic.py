import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):

    def __init__(self, num_inputs, num_actions):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 100)
        self.state_value = nn.Linear(100, 1)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        value = self.state_value(x)
        return value


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'num_inputs': 4, 'num_actions': 4}]
