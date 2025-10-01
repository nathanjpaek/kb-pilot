import torch
from torch import nn


class LinearRegression(nn.Module):

    def __init__(self, in_features: 'int', out_features: 'int', bias:
        'bool'=True):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = bias
        if bias:
            self.bias_term = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        x = x @ self.weights.t()
        if self.bias:
            x += self.bias_term
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
