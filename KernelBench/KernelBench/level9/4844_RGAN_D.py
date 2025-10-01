import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader as DataLoader


class RGAN_D(nn.Module):

    def __init__(self, in_size, hidden_size, num_outcomes):
        super(RGAN_D, self).__init__()
        self.L1 = nn.Linear(in_size, hidden_size)
        self.L2 = nn.Linear(hidden_size, hidden_size)
        self.L3 = nn.Linear(hidden_size, hidden_size)
        self.L4 = nn.Linear(hidden_size, num_outcomes)

    def forward(self, x):
        out = self.L1(x)
        out = F.leaky_relu(out, 0.02)
        out = self.L2(out)
        out = F.leaky_relu(out, 0.02)
        out = self.L3(out)
        out = F.leaky_relu(out, 0.02)
        out = self.L4(out)
        out = F.softmax(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_size': 4, 'hidden_size': 4, 'num_outcomes': 4}]
