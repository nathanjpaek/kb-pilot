import torch
import torch.nn as nn
from torch.nn import functional as F


class simple_mlp(nn.Module):

    def __init__(self, feature_dim, layer, hidden):
        super(simple_mlp, self).__init__()
        self.layer = layer
        self.linear1 = nn.Linear(feature_dim, hidden)
        if layer == 2:
            self.linear2 = nn.Linear(hidden, hidden)
        self.linear3 = nn.Linear(hidden, 1)

    def forward(self, x, weights=None):
        hidden = F.relu(self.linear1(x))
        if self.layer == 2:
            hidden = F.relu(self.linear2(hidden))
        out = self.linear3(hidden)
        return out, hidden


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'feature_dim': 4, 'layer': 1, 'hidden': 4}]
