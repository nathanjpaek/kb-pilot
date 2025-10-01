import torch
import torch.nn.functional as F
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, n_in, n_units, n_out):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(n_in, n_units)
        self.l2 = nn.Linear(n_units, n_units)
        self.l3 = nn.Linear(n_units, n_out)

    def forward(self, x):
        x = x.view((len(x), -1))
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return F.log_softmax(self.l3(h2), dim=1)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'n_in': 4, 'n_units': 4, 'n_out': 4}]
