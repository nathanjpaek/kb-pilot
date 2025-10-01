import torch
import numpy as np
from torch import nn
import torch.autograd


def fanin_(size):
    fan_in = size[0]
    weight = 1.0 / np.sqrt(fan_in)
    return torch.Tensor(size).uniform_(-weight, weight)


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim, h1=64, h2=32, init_w=0.003):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(state_dim, h1)
        self.linear1.weight.data = fanin_(self.linear1.weight.data.size())
        self.linear2 = nn.Linear(h1 + action_dim, h2)
        self.linear2.weight.data = fanin_(self.linear2.weight.data.size())
        self.linear3 = nn.Linear(h2, 1)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.relu = nn.ReLU()

    def forward(self, state, action):
        x = self.linear1(state)
        x = self.relu(x)
        x = self.linear2(torch.cat([x, action], 1))
        x = self.relu(x)
        x = self.linear3(x)
        return x


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'state_dim': 4, 'action_dim': 4}]
