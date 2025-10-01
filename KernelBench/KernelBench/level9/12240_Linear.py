from torch.nn import Module
import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import torch.nn.modules.loss


class Linear(Module):
    """
    to embedding feature
    """

    def __init__(self, in_features, out_features, dropout=0.0, act=F.relu,
        bias=True, sparse_inputs=False, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.bias = bias
        self.sparse_inputs = sparse_inputs
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()
        if self.bias:
            self.weight_bias = Parameter(torch.FloatTensor(torch.zeros(
                out_features)))

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input):
        if self.sparse_inputs:
            output = torch.spmm(input, self.weight)
        else:
            output = torch.mm(input, self.weight)
        if self.bias:
            output += self.weight_bias
        return self.act(output)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
