import torch
from torch import nn as nn
from torch.nn.parameter import Parameter


class EnergyEstimateWidthRescale(nn.Module):

    def __init__(self, scales):
        super(EnergyEstimateWidthRescale, self).__init__()
        self.scales = Parameter(torch.tensor(scales, dtype=torch.float32),
            requires_grad=False)

    def forward(self, x):
        assert x.dim() != 1
        x = x / self.scales
        return torch.cat([(x[:, 0].detach() * x[:, 1]).unsqueeze(1), x[:, 1
            :-2] * x[:, 2:-1], (x[:, -2] * x[:, -1].detach()).unsqueeze(1)],
            dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'scales': 1.0}]
