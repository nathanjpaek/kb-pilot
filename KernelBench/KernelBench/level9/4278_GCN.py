from torch.nn import Module
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F


class GCLayer(Module):

    def __init__(self, dim_in, dim_out):
        super(GCLayer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.weight = Parameter(torch.FloatTensor(self.dim_in, self.dim_out))
        self.bias = Parameter(torch.FloatTensor(self.dim_out))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        return output + self.bias


class GCN(nn.Module):

    def __init__(self, features, hidden, classes, dropout, layers=2):
        super(GCN, self).__init__()
        self.gc1 = GCLayer(features, hidden)
        self.gc2 = GCLayer(hidden, classes)
        self.gc3 = None
        self.dropout = dropout
        if layers == 3:
            self.gc2 = GCLayer(hidden, hidden)
            self.gc3 = GCLayer(hidden, classes)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout)
        x = self.gc2(x, adj)
        if self.gc3 is not None:
            x = self.gc3(x, adj)
        return F.log_softmax(x, dim=1)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'features': 4, 'hidden': 4, 'classes': 4, 'dropout': 0.5}]
