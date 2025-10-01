import torch
import torch.nn.functional as F
from torch import nn


class Linear2(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout):
        super(Linear2, self).__init__()
        self.linear1 = nn.Linear(nfeat, nhid, bias=True)
        self.linear2 = nn.Linear(nhid, nclass, bias=True)
        self.dropout = dropout

    def forward(self, x, adj=None):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.linear1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear2(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nfeat': 4, 'nhid': 4, 'nclass': 4, 'dropout': 0.5}]
