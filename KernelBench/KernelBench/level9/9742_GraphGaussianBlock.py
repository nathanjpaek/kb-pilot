import math
import torch
import torch.nn as nn
import torch.optim
import torch.multiprocessing
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True, node_n=48,
        out_node_n=None):
        super(GraphConvolution, self).__init__()
        if out_node_n is None:
            out_node_n = node_n
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(out_node_n, node_n))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.att, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features
            ) + ' -> ' + str(self.out_features) + ')'


class GraphGaussianBlock(nn.Module):

    def __init__(self, in_nodes, in_features, n_z_nodes, n_z_features):
        """
        :param input_feature: num of input feature
        :param n_z: dim of distribution
        """
        super(GraphGaussianBlock, self).__init__()
        self.in_features = in_features
        self.in_nodes = in_nodes
        self.n_z_features = n_z_features
        self.n_z_nodes = n_z_nodes
        self.z_mu_graphconv = GraphConvolution(in_features, n_z_features,
            bias=True, node_n=in_nodes, out_node_n=n_z_nodes)
        self.z_log_var_graphconv = GraphConvolution(in_features,
            n_z_features, bias=True, node_n=in_nodes, out_node_n=n_z_nodes)

    def forward(self, x):
        y = x
        mu = self.z_mu_graphconv(y)
        log_var = self.z_log_var_graphconv(y)
        log_var = torch.clamp(log_var, min=-20.0, max=3.0)
        return mu, log_var


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_nodes': 4, 'in_features': 4, 'n_z_nodes': 4,
        'n_z_features': 4}]
