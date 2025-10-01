import torch
import torch.nn as nn


class Net(nn.Module):

    def __init__(self, n_inputs, n_units=[50, 50, 50]):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_inputs, n_units[0])
        self.fc2 = nn.Linear(n_units[0], n_units[1])
        self.fc3 = nn.Linear(n_units[1], n_units[2])
        self.out = nn.Linear(n_units[2], 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return self.out(x)

    def basis_funcs(self, x, bias=False, linear=False):
        raw_x = x
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        if linear:
            x = torch.cat((x, raw_x), dim=-1)
        if bias:
            x = torch.cat((x, torch.ones(size=(raw_x.shape[0], 1))), dim=-1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_inputs': 4}]
