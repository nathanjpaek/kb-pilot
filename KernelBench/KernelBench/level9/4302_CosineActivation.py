import torch
import torch.nn as nn


def t2v(tau, f, weight_linear, bias_linear, weight_periodic, bias_periodic,
    arg=None):
    if arg:
        v1 = f(torch.matmul(tau, weight_linear) + bias_linear, arg)
    else:
        v1 = f(torch.matmul(tau, weight_linear) + bias_linear)
    v2 = torch.matmul(tau, weight_periodic) + bias_periodic
    return torch.cat([v1, v2], -1)


class CosineActivation(nn.Module):

    def __init__(self, in_features, output_features):
        super(CosineActivation, self).__init__()
        self.output_features = output_features
        self.weight_linear = nn.parameter.Parameter(torch.randn(in_features,
            output_features))
        self.bias_linear = nn.parameter.Parameter(torch.randn(output_features))
        self.weight_periodic = nn.parameter.Parameter(torch.randn(
            in_features, output_features))
        self.bias_periodic = nn.parameter.Parameter(torch.randn(
            output_features))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.weight_linear, self.bias_linear, self.
            weight_periodic, self.bias_periodic)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'output_features': 4}]
