import torch
import torch.nn as nn
import torch.nn.functional as F


class ANN(nn.Module):

    def __init__(self, n_li, n_l1, n_l2, n_l3, n_lo):
        super(ANN, self).__init__()
        self.lin_in = nn.Linear(n_li, n_l1)
        self.lin_h1 = nn.Linear(n_l1, n_l2)
        self.lin_h2 = nn.Linear(n_l2, n_l3)
        self.lin_out = nn.Linear(n_l3, n_lo)

    def forward(self, inputs):
        out = F.relu(self.lin_in(inputs))
        out = F.relu(self.lin_h1(out))
        out = F.relu(self.lin_h2(out))
        out = F.sigmoid(self.lin_out(out))
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_li': 4, 'n_l1': 4, 'n_l2': 4, 'n_l3': 4, 'n_lo': 4}]
