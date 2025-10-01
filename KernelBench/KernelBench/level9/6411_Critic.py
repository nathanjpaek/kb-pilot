import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(fan_in)
    return -lim, lim


class Critic(nn.Module):

    def __init__(self, critic_in, action_size, seed, fc1_units=512,
        fc2_units=256):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(critic_in, fc1_units)
        self.fc2 = nn.Linear(fc1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-0.003, 0.003)

    def forward(self, states, actions):
        xs = F.relu(self.fc1(states))
        x = torch.cat((xs, actions), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'critic_in': 4, 'action_size': 4, 'seed': 4}]
