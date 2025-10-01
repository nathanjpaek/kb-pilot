import torch
import torch.nn as nn


class DQMLP(nn.Module):

    def __init__(self, n_observations, n_actions, n_hidden):
        super().__init__()
        self.linear = nn.Linear(n_observations, n_hidden)
        self.linear_adv = nn.Linear(n_hidden, n_actions)
        self.linear_value = nn.Linear(n_hidden, 1)
        self.n_actions = n_actions

    def forward_common(self, frame):
        z = torch.tanh(self.linear(frame))
        return z

    def forward_value(self, z):
        return self.linear_value(z)

    def forward_advantage(self, z):
        adv = self.linear_adv(z)
        advm = adv.mean(1).unsqueeze(-1).repeat(1, self.n_actions)
        return adv - advm

    def forward(self, state):
        z = self.forward_common(state)
        v = self.forward_value(z)
        adv = self.forward_advantage(z)
        return v + adv


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'n_observations': 4, 'n_actions': 4, 'n_hidden': 4}]
