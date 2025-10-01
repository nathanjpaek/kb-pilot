import math
import torch
import torch.nn


class SirenModule(torch.nn.Module):

    def __init__(self, in_features, out_features, weight_multiplier=1.0):
        super(SirenModule, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        init_bounds = math.sqrt(6 / in_features) * weight_multiplier
        torch.nn.init.uniform_(self.linear.weight, a=-init_bounds, b=
            init_bounds)

    def forward(self, x):
        return torch.sin(self.linear(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
