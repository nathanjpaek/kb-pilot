import torch
import torch.nn as nn
import torch.nn.functional as F


class FC2LayersShortcut(nn.Module):

    def __init__(self, n_in, n_hidden, n_out, activation=F.relu):
        super(FC2LayersShortcut, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden + n_in, n_out)
        self.activation = activation

    def forward(self, x):
        h = self.activation(self.fc1(x))
        h = torch.cat((h, x), 1)
        x = self.fc2(h)
        return x


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'n_in': 4, 'n_hidden': 4, 'n_out': 4}]
