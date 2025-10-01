from torch.nn import Module
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class sparse_dropout(Module):
    """
    Sparse dropout implementation
    """

    def __init__(self):
        super(sparse_dropout, self).__init__()

    def forward(self, input, p=0.5):
        if self.training is True and p > 0.0:
            random = torch.rand_like(input._values())
            mask = random.ge(p)
            new_indices = torch.masked_select(input._indices(), mask).reshape(
                2, -1)
            new_values = torch.masked_select(input._values(), mask)
            output = torch.sparse.FloatTensor(new_indices, new_values,
                input.shape)
            return output
        else:
            return input


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, node_dropout=0.0,
        edge_dropout=0.0, bias=True):
        super(GraphConvolution, self).__init__()
        self.sparse_dropout = sparse_dropout()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.zeros(in_features, out_features))
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        nn.init.uniform_(self.weight, -stdv, stdv)
        if bias:
            self.bias = Parameter(torch.zeros(1, out_features))
            nn.init.uniform_(self.bias, -stdv, stdv)
        else:
            self.register_parameter('bias', None)
        self.node_dropout = node_dropout
        self.edge_dropout = edge_dropout

    def forward(self, input, adj):
        adj = self.sparse_dropout(adj, self.edge_dropout)
        support = torch.mm(input, self.weight)
        support = F.dropout(support, self.node_dropout, training=self.training)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features
            ) + ' -> ' + str(self.out_features) + ')'


class GCN1(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN1, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc1(x, adj)
        return x


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'nfeat': 4, 'nhid': 4, 'nclass': 4, 'dropout': 0.5}]
