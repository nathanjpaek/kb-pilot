import torch
import torch.nn.functional as F
from torch import nn


class Model(nn.Module):

    def __init__(self, n_input: 'int', state_dict=None):
        super(Model, self).__init__()
        self.n_input = n_input
        self.fc = nn.Linear(n_input, 20)
        self.output = nn.Linear(20, 1)
        nn.init.xavier_uniform_(self.fc.weight, gain=nn.init.calculate_gain
            ('relu'))
        nn.init.xavier_uniform_(self.output.weight, gain=nn.init.
            calculate_gain('relu'))
        if state_dict is not None:
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = F.relu(self.fc(x))
        return self.output(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_input': 4}]
