import torch
import torch.nn as nn
import torch.nn.functional as F


class PositiveLinear(nn.Linear):

    def forward(self, input):
        return F.linear(input, self.weight ** 2, self.bias)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
