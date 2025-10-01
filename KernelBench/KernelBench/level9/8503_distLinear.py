import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import WeightNorm


class distLinear(nn.Module):

    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear(indim, outdim, bias=False)
        self.class_wise_learnable_norm = True
        if self.class_wise_learnable_norm:
            WeightNorm.apply(self.L, 'weight', dim=0)
        if outdim <= 200:
            self.scale_factor = 2
        else:
            self.scale_factor = 10

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 1e-05)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim=1).unsqueeze(1
                ).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 1e-05)
        cos_dist = self.L(x_normalized)
        scores = self.scale_factor * cos_dist
        return scores


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'indim': 4, 'outdim': 4}]
