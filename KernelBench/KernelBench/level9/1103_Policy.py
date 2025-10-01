import torch
from torch import nn
from torch.nn import functional as F


class Policy(nn.Module):

    def __init__(self, act_dim, obs_dim):
        super(Policy, self).__init__()
        self.fc0 = nn.Linear(act_dim, 128)
        self.fc1 = nn.Linear(128, obs_dim)

    def forward(self, x):
        x = x.type_as(self.fc0.bias)
        x = F.relu(self.fc0(x))
        x = F.dropout(x, training=self.training)
        x = self.fc1(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'act_dim': 4, 'obs_dim': 4}]
