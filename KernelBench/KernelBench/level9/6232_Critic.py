import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


def hidden_unit(layer):
    inp = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(inp)
    return -lim, lim


class Critic(nn.Module):

    def __init__(self, state_size, action_size, seed=2, fc1_units=256,
        fc2_units=256, fc3_units=128):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)
        self.reset_weights()

    def reset_weights(self):
        self.fc1.weight.data.uniform_(*hidden_unit(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_unit(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_unit(self.fc3))
        self.fc4.weight.data.uniform_(-0.003, 0.003)

    def forward(self, state, action):
        x = F.leaky_relu(self.fc1(state))
        x = torch.cat((x, action), dim=1)
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return self.fc4(x)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'state_size': 4, 'action_size': 4}]
