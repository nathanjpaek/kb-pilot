import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class GraphConv(nn.Module):

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConv, self).__init__()
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

    def forward(self, x, adj, D=None):
        if x.dim() == 3:
            xw = torch.matmul(x, self.weight)
            output = torch.bmm(adj, xw)
        elif x.dim() == 2:
            xw = torch.mm(x, self.weight)
            output = torch.spmm(adj, xw)
        if D is not None:
            output = output * 1.0 / D
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features
            ) + ' -> ' + str(self.out_features) + ')'


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, dropout=0.0):
        super(BasicBlock, self).__init__()
        self.gc = GraphConv(inplanes, planes)
        self.relu = nn.ReLU(inplace=True)
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

    def forward(self, x, adj, D=None):
        x = self.gc(x, adj, D)
        x = self.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'inplanes': 4, 'planes': 4}]
