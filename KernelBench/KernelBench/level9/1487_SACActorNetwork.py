import torch
import torch.nn.functional as F
import torch.nn as nn


class SACActorNetwork(nn.Module):

    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(SACActorNetwork, self).__init__()
        n_input = input_shape[-1]
        n_output = output_shape[0]
        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)
        nn.init.xavier_uniform_(self._h1.weight, gain=nn.init.
            calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight, gain=nn.init.
            calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight, gain=nn.init.
            calculate_gain('linear'))

    def forward(self, state):
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        a = self._h3(features2)
        return a


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_shape': [4, 4], 'output_shape': [4, 4],
        'n_features': 4}]
