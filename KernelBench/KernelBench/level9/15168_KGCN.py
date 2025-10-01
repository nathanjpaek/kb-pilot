from torch.nn import Module
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features
            ) + ' -> ' + str(self.out_features) + ')'


class KGCN(Module):
    """
    A bit more complex GNN to deal with non-convex feature space.
    """

    def __init__(self, nhidden, nfeat, nclass, degree):
        super(KGCN, self).__init__()
        self.Wx = GraphConvolution(nfeat, nhidden)
        self.W = nn.Linear(nhidden, nclass)
        self.d = degree

    def forward(self, x, adj):
        h = F.relu(self.Wx(x, adj))
        for i in range(self.d):
            h = torch.spmm(adj, h)
        return self.W(h)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'nhidden': 4, 'nfeat': 4, 'nclass': 4, 'degree': 4}]
