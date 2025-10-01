import torch
from torch import nn
from torch.nn import functional as F
from typing import *


class MLPEncoder(torch.nn.Module):

    def __init__(self, indim, hiddim, outdim):
        super(MLPEncoder, self).__init__()
        self.fc = nn.Linear(indim, hiddim)
        self.fc2 = nn.Linear(hiddim, 2 * outdim)
        self.outdim = outdim

    def forward(self, x):
        output = self.fc(x)
        output = F.relu(output)
        output = self.fc2(output)
        return output[:, :self.outdim], output[:, self.outdim:]


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'indim': 4, 'hiddim': 4, 'outdim': 4}]
