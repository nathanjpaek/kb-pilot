import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):

    def __init__(self, input_features, output_features, bias=False):
        super(GCNLayer, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weights = nn.Parameter(torch.FloatTensor(input_features,
            output_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, adj, x):
        support = torch.mm(x, self.weights)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        return output


class GCN(nn.Module):

    def __init__(self, input_size, hidden_size, num_class, dropout, bias=False
        ):
        super(GCN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_class = num_class
        self.gcn1 = GCNLayer(input_size, hidden_size, bias=bias)
        self.gcn2 = GCNLayer(hidden_size, num_class, bias=bias)
        self.dropout = dropout

    def forward(self, adj, x):
        x = F.relu(self.gcn1(adj, x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn2(adj, x)
        return F.log_softmax(x, dim=1)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4, 'num_class': 4,
        'dropout': 0.5}]
