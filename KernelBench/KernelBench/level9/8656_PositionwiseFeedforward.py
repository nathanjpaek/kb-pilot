import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedforward(nn.Module):

    def __init__(self, hid_dim: 'int', pf_dim: 'int', dropout: 'float'):
        super().__init__()
        self.hid_dim = hid_dim
        self.pf_dim = pf_dim
        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: 'Tensor'):
        x = x.permute(0, 2, 1)
        x = self.dropout(F.relu(self.fc_1(x)))
        x = self.fc_2(x)
        x = x.permute(0, 2, 1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'hid_dim': 4, 'pf_dim': 4, 'dropout': 0.5}]
