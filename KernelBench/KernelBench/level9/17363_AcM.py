import torch
from torch import nn


class AcM(nn.Module):

    def __init__(self, in_dim: 'int', ac_dim: 'int', ac_lim: 'int',
        discrete: 'bool'=True):
        super().__init__()
        self.ac_lim = ac_lim
        self.discrete = discrete
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, ac_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        if self.discrete:
            x = torch.softmax(self.fc3(x), dim=1)
        else:
            x = torch.tanh(self.fc3(x))
            x = x * self.ac_lim
        return x

    def act(self, obs):
        action = self.forward(obs)
        if self.discrete:
            action = torch.argmax(action, dim=1)
        return action


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'ac_dim': 4, 'ac_lim': 4}]
