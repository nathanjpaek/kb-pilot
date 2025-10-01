import torch
import torch.nn as nn
from collections.abc import Iterable


class MLPArchitecture(nn.Module):

    def __init__(self, batch_size, n_outputs, state_size):
        super(MLPArchitecture, self).__init__()
        if isinstance(state_size, Iterable):
            assert len(state_size) == 1
            state_size = state_size[0]
        self.batch_size = batch_size
        self.n_outputs = n_outputs
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, n_outputs)

    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        out = self.fc3(h)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'batch_size': 4, 'n_outputs': 4, 'state_size': 4}]
