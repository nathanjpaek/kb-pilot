import torch
from math import *
import torch.nn as nn
import torch.nn.functional as F


class Stalin3000_anal_probe(nn.Module):

    def __init__(self, n):
        super(Stalin3000_anal_probe, self).__init__()
        self.n = n
        self.insider = nn.Linear(n, n + 2)
        self.hl1 = nn.Linear(n + 2, n + 2)
        self.hl2 = nn.Linear(n + 2, int(n / 2))
        self.outsider = nn.Linear(int(n / 2), 1)

    def forward(self, x):
        x = F.dropout(F.relu(self.insider(x)), p=0.5)
        x = F.dropout(F.relu(self.hl1(x)), p=0.5)
        x = F.dropout(F.relu(self.hl2(x)), p=0.5)
        x = F.dropout(F.relu(self.outsider(x)), p=0.5)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n': 4}]
