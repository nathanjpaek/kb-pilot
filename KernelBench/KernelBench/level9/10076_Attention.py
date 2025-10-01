import torch
from torch import nn


class Attention(nn.Module):

    def __init__(self, feature_dim, K, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.supports_masking = True
        self.bias = bias
        self.feature_dim = feature_dim
        self.K = K
        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        if bias:
            self.b = nn.Parameter(torch.zeros(K))

    def forward(self, x):
        B, N, K, feature_dim = x.shape
        eij = torch.mm(x.contiguous().view(-1, feature_dim), self.weight).view(
            -1, K)
        if self.bias:
            eij = eij + self.b
        eij = torch.tanh(eij)
        a = torch.exp(eij)
        a = a / torch.sum(a, 1, keepdim=True) + 1e-10
        weighted_input = x * a.view(B, N, K, 1)
        return torch.sum(weighted_input, 2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'feature_dim': 4, 'K': 4}]
