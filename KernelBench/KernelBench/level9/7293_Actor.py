import torch
import numpy as np
from torch import nn
import torch.autograd


def fanin_(size):
    fan_in = size[0]
    weight = 1.0 / np.sqrt(fan_in)
    return torch.Tensor(size).uniform_(-weight, weight)


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, h1=64, h2=32, init_w=0.003):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(state_dim, h1)
        self.linear1.weight.data = fanin_(self.linear1.weight.data.size())
        self.linear2 = nn.Linear(h1, h2)
        self.linear2.weight.data = fanin_(self.linear2.weight.data.size())
        self.linear3 = nn.Linear(h2, action_dim)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if cuda else 'cpu')

    def forward(self, state):
        x = self.linear1(state)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.tanh(x)
        return x

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0]


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_dim': 4, 'action_dim': 4}]
