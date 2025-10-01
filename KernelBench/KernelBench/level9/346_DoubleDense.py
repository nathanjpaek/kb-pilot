import torch
from torch import nn
from torch.nn import functional as F


class DoubleDense(nn.Module):

    def __init__(self, in_channels, hidden_neurons, output_channels):
        super(DoubleDense, self).__init__()
        self.dense1 = nn.Linear(in_channels, out_features=hidden_neurons)
        self.dense2 = nn.Linear(in_features=hidden_neurons, out_features=
            hidden_neurons // 2)
        self.dense3 = nn.Linear(in_features=hidden_neurons // 2,
            out_features=output_channels)

    def forward(self, x):
        out = F.relu(self.dense1(x.view(x.size(0), -1)))
        out = F.relu(self.dense2(out))
        out = self.dense3(out)
        return out


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'hidden_neurons': 4, 'output_channels': 4}]
