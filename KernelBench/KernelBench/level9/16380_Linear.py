import torch
import torch.nn.functional as F
import torch.nn as nn


class Linear(nn.Linear):

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True)
        weight = weight - weight_mean
        std = weight.std(dim=1, keepdim=True) + 1e-05
        weight = weight / std.expand_as(weight)
        return F.linear(x, weight, self.bias)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
