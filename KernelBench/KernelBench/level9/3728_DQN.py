import torch
import torch.nn as nn


class DQN(nn.Module):

    def __init__(self, obs_size: 'int', num_actions: 'int', hidden_size:
        'int'=20):
        super(DQN, self).__init__()
        self.l1 = nn.Linear(obs_size, hidden_size)
        self.n1 = nn.LayerNorm(hidden_size, elementwise_affine=True)
        self.l3 = nn.Linear(hidden_size, num_actions)
        self.activ = torch.nn.LeakyReLU()

    def forward(self, x):
        hidden = self.activ(self.n1(self.l1(x)))
        output = self.l3(hidden)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'obs_size': 4, 'num_actions': 4}]
