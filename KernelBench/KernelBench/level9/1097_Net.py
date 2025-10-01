import torch
from torch import nn
from torch.nn import functional as F


class Net(nn.Module):

    def __init__(self, obs_dim, act_dim):
        super(Net, self).__init__()
        self.fc0 = nn.Linear(obs_dim, 128)
        self.fc1 = nn.Linear(128, act_dim)

    def forward(self, x):
        x = x.type_as(self.fc0.bias)
        x = F.relu(self.fc0(x))
        x = self.fc1(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'obs_dim': 4, 'act_dim': 4}]
