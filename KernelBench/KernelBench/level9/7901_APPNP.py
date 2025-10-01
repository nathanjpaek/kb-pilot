import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):

    def __init__(self, in_features, out_features, dropout, bias=False):
        super(Linear, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            stdv = 1.0 / math.sqrt(self.weight.size(1))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        input = F.dropout(input, self.dropout, training=self.training)
        output = torch.matmul(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class APPNP(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout, K, alpha):
        super(APPNP, self).__init__()
        self.Linear1 = Linear(nfeat, nhid, dropout, bias=False)
        self.Linear2 = Linear(nhid, nclass, dropout, bias=False)
        self.alpha = alpha
        self.K = K

    def forward(self, x, adj):
        x = torch.relu(self.Linear1(x))
        h0 = self.Linear2(x)
        h = h0
        for _ in range(self.K):
            h = (1 - self.alpha) * torch.matmul(adj, h) + self.alpha * h0
        return torch.log_softmax(h, dim=-1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nfeat': 4, 'nhid': 4, 'nclass': 4, 'dropout': 0.5, 'K': 4,
        'alpha': 4}]
