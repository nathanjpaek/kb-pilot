import torch
import torch.nn as nn


class HighwayFC(nn.Module):

    def __init__(self, indim, outdim, activation='relu', bias=-1):
        super(HighwayFC, self).__init__()
        self.indim = indim
        self.outdim = outdim
        if activation == 'selu':
            self.activation = nn.SELU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()
        self.fc = nn.Linear(self.indim, self.outdim)
        self.gate = nn.Linear(self.indim, self.outdim)
        self.gateact = nn.Sigmoid()
        self.gate.bias.data.fill_(bias)

    def forward(self, x):
        H = self.activation(self.fc(x))
        T = self.gateact(self.gate(x))
        out = H * T + x * (1 - T)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'indim': 4, 'outdim': 4}]
