from torch.nn import Module
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, init_method=
        'xavier'):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters(method=init_method)

    def reset_parameters(self, method='xavier'):
        if method == 'uniform':
            stdv = 1.0 / math.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)
            if self.bias is not None:
                self.bias.data.uniform_(-stdv, stdv)
        elif method == 'kaiming':
            nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')
            if self.bias is not None:
                nn.init.constant_(self.bias.data, 0.0)
        elif method == 'xavier':
            nn.init.xavier_normal_(self.weight.data, gain=0.02)
            if self.bias is not None:
                nn.init.constant_(self.bias.data, 0.0)
        else:
            raise NotImplementedError

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


class GCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout, init_method='xavier',
        dropout_input=False):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid, init_method=init_method)
        self.gc2 = GraphConvolution(nhid, nclass, init_method=init_method)
        self.dropout = dropout
        self.dropout_input = dropout_input

    def bottleneck(self, path1, path2, path3, adj, in_x):
        return F.relu(path3(F.relu(path2(F.relu(path1(in_x, adj)), adj)), adj))

    def forward(self, x, adj):
        if self.dropout_input:
            x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'nfeat': 4, 'nhid': 4, 'nclass': 4, 'dropout': 0.5}]
