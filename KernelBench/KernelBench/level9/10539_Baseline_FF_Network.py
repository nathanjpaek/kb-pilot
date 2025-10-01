import torch
from torch import nn
import torch.nn.functional as F


class Baseline_FF_Network(nn.Module):

    def __init__(self):
        super().__init__()
        h1_dim = 500
        h2_dim = 500
        self.fc1 = nn.Linear(4, h1_dim)
        self.fc2 = nn.Linear(h1_dim, h2_dim)
        self.fc3 = nn.Linear(h1_dim, h2_dim)
        self.fc_last = nn.Linear(h2_dim, 4)

    def forward(self, qqd):
        x = F.softplus(self.fc1(qqd))
        x = F.softplus(self.fc2(x))
        x = F.softplus(self.fc3(x))
        x = self.fc_last(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
