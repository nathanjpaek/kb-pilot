from torch.nn import Module
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, topology, bsize, i, n_class, bias=True):
        super(GraphConvolution, self).__init__()
        if i == 0:
            self.in_features = in_features
        else:
            self.in_features = topology[i - 1] * bsize
        if i == len(topology):
            self.out_features = n_class
        else:
            self.out_features = topology[i] * bsize
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.
            out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(self.out_features))
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


class GCN(nn.Module):

    def __init__(self, infeat, bsize, topology, n_class, dropout):
        super(GCN, self).__init__()
        self.num_layers = len(topology)
        self.layers = nn.ModuleDict({'gc{}'.format(i): GraphConvolution(
            infeat, topology, bsize, i, n_class) for i in range(self.
            num_layers)})
        self.outlayer = GraphConvolution(infeat, topology, bsize, self.
            num_layers, n_class)
        self.dropout = dropout

    def forward(self, x, adj, ls=False):
        for i in range(self.num_layers):
            x = self.layers['gc' + str(i)](x, adj)
            x = F.relu(x)
            if i == 0:
                x = F.dropout(x, self.dropout, training=self.training)
        if ls:
            pred = x
        else:
            x = self.outlayer(x, adj)
            pred = F.log_softmax(x, dim=1)
        return pred


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'infeat': 4, 'bsize': 4, 'topology': [4, 4], 'n_class': 4,
        'dropout': 0.5}]
