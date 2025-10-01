import torch
from torch import nn


class PositiveLinear(nn.Module):

    def __init__(self, in_features: 'int', out_features: 'int') ->None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.softplus = nn.Softplus()

    def forward(self, input: 'torch.Tensor'):
        return input @ self.softplus(self.weight) + self.softplus(self.bias)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
