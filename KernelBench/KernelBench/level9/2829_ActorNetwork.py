import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):

    def __init__(self, input_shape, output_shape, **kwargs):
        super(ActorNetwork, self).__init__()
        n_input = input_shape[-1]
        n_output = output_shape[0]
        self._h = nn.Linear(n_input, n_output)
        nn.init.xavier_uniform_(self._h.weight, gain=nn.init.calculate_gain
            ('relu'))

    def forward(self, state):
        return F.relu(self._h(torch.squeeze(state, 1).float()))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_shape': [4, 4], 'output_shape': [4, 4]}]
