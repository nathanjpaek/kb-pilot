import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import *


class GateLayer(nn.Module):

    def __init__(self, dim, target_dim=None, dropout=None):
        super(GateLayer, self).__init__()
        if target_dim is None:
            target_dim = dim
            self.linear_transform = False
        else:
            self.target_dim = target_dim
            self.linear_transform = True
        self.gate = nn.Conv1d(dim, target_dim, 1)
        if self.linear_transform:
            self.linear = nn.Conv1d(dim, target_dim, 1)
        self.dropout = dropout

    def forward(self, x):
        tx = x.transpose(1, 2)
        gate = F.sigmoid(self.gate(tx))
        if self.linear_transform:
            linear = self.linear(tx)
        else:
            linear = tx
        res = (gate * linear).transpose(2, 1)
        if self.dropout:
            res = self.dropout(res)
        return res


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
