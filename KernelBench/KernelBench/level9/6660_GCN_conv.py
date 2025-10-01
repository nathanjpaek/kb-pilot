import math
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GCN_conv(nn.Module):

    def __init__(self, in_ft, out_ft, bias=False, dropout=0.0, activation=F
        .relu):
        super(GCN_conv, self).__init__()
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.dropout = nn.Dropout(p=dropout)
        self.activation = activation

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: 'torch.Tensor', G: 'torch.Tensor'):
        x = self.dropout(x)
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return self.activation(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_ft': 4, 'out_ft': 4}]
