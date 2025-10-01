import torch
import torch.nn as nn
import torch.utils


class Net(nn.Module):

    def __init__(self, n_inputs, n_units=50):
        super(Net, self).__init__()
        self.fc = nn.Linear(n_inputs, n_units)
        self.out = nn.Linear(n_units, 1)

    def forward(self, x):
        x = torch.tanh(self.fc(x))
        return torch.sigmoid(self.out(x))

    def basis_funcs(self, x):
        return torch.tanh(self.fc(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_inputs': 4}]
