import math
import torch


class CoSirenModule(torch.nn.Module):

    def __init__(self, in_features, out_features, weight_multiplier=1.0):
        super(CoSirenModule, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features // 2)
        init_bounds = math.sqrt(24 / in_features) * weight_multiplier
        torch.nn.init.uniform_(self.linear.weight, a=-init_bounds, b=
            init_bounds)

    def forward(self, x):
        x = self.linear(x)
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1) - math.pi / 4


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
