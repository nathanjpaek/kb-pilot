from torch.nn import Module
import math
import torch
import torch.nn as nn
import torch.utils.data
import torch
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GCN_Spectral(Module):
    """ Simple GCN layer, similar to https://arxiv.org/abs/1609.02907 """

    def __init__(self, in_units: 'int', out_units: 'int', bias: 'bool'=True
        ) ->None:
        super(GCN_Spectral, self).__init__()
        self.in_units = in_units
        self.out_units = out_units
        self.weight = Parameter(torch.FloatTensor(in_units, out_units))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_units))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) ->None:
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input: 'torch.Tensor', adj: 'torch.Tensor'
        ) ->torch.Tensor:
        """

        weight=(input_dim X hid_dim)
        :param input: (#samples X input_dim)
        :param adj:
        :return:
        """
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_units
            ) + ' -> ' + str(self.out_units) + ')'


class GCN(nn.Module):

    def __init__(self, nfeat: 'int', nhid: 'int', nclass: 'int', dropout:
        'float') ->None:
        super(GCN, self).__init__()
        self.gc1 = GCN_Spectral(nfeat, nhid)
        self.gc2 = GCN_Spectral(nhid, nfeat)
        self.dropout = dropout

    def forward(self, x: 'torch.Tensor', adj: 'torch.Tensor') ->(torch.
        Tensor, torch.Tensor):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1), x


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'nfeat': 4, 'nhid': 4, 'nclass': 4, 'dropout': 0.5}]
