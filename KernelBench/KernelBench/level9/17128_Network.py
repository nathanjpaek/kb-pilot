import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):

    def __init__(self, input_dim):
        super(Network, self).__init__()
        self.first_layer = nn.Linear(input_dim, 6)
        self.out_layer = nn.Linear(6, 1)

    def forward(self, x):
        out = self.first_layer(x)
        out = F.relu(out)
        out = self.out_layer(out)
        out = F.relu(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4}]
