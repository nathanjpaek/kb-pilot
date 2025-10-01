import math
import torch
from torch import nn


class FiLMSIREN(nn.Module):

    def __init__(self, in_features: 'int', out_features: 'int', omega_0:
        'float'=30.0, is_first: 'bool'=False, bias: 'bool'=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.is_first = is_first
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            b = math.sqrt(6.0 / self.in_features
                ) if self.is_first else math.sqrt(6.0 / self.in_features
                ) / self.omega_0
            self.linear.weight.uniform_(-b, b)

    def forward(self, x, gamma=None, beta=None):
        out = self.linear(x)
        if gamma is not None:
            out = out * gamma
        if beta is not None:
            out = out + beta
        out = torch.sin(self.omega_0 * out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
