import torch
from torch import nn


class QNetwork(nn.Module):

    def __init__(self, num_states, num_actions):
        super().__init__()
        self._num_states = num_states
        self._num_actions = num_actions
        self._fc1 = nn.Linear(self._num_states, 100)
        self._relu1 = nn.ReLU(inplace=True)
        self._fc2 = nn.Linear(100, 60)
        self._relu2 = nn.ReLU(inplace=True)
        self._fc_final = nn.Linear(60, self._num_actions)
        nn.init.zeros_(self._fc1.bias)
        nn.init.zeros_(self._fc2.bias)
        nn.init.zeros_(self._fc_final.bias)
        nn.init.uniform_(self._fc_final.weight, a=-1e-06, b=1e-06)

    def forward(self, state):
        h = self._relu1(self._fc1(state))
        h = self._relu2(self._fc2(h))
        q_values = self._fc_final(h)
        return q_values


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_states': 4, 'num_actions': 4}]
