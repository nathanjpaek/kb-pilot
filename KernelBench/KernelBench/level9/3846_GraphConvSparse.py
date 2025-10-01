import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial)


class GraphConvSparse(nn.Module):

    def __init__(self, input_dim, output_dim, activation=F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim)
        self.activation = activation

    def forward(self, features, adj):
        _b, _n, _d = features.shape
        x = features
        x = torch.einsum('bnd,df->bnf', (x, self.weight))
        x = torch.bmm(adj, x)
        outputs = self.activation(x)
        return outputs


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4}]
