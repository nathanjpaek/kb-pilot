import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):

    def __init__(self, statedim, actiondim, hiddendim, init_w=0.0003):
        super().__init__()
        self.linear1 = nn.Linear(statedim + actiondim, hiddendim)
        self.linear2 = nn.Linear(hiddendim, hiddendim)
        self.linear3 = nn.Linear(hiddendim, 1)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        stateact = torch.cat([state, action], dim=-1)
        x = F.relu(self.linear1(stateact))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'statedim': 4, 'actiondim': 4, 'hiddendim': 4}]
