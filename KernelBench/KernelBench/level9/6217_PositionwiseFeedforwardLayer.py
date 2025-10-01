import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedforwardLayer(nn.Module):

    def __init__(self, hid_dim: 'int', pf_dim: 'int', dropout: 'float') ->None:
        super().__init__()
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: 'torch.FloatTensor') ->torch.FloatTensor:
        x = F.relu(self.fc_1(x)) ** 2
        x = self.fc_2(x)
        x = self.dropout(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hid_dim': 4, 'pf_dim': 4, 'dropout': 0.5}]
