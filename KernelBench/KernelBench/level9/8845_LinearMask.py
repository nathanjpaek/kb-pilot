import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F


class LinearMask(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(LinearMask, self).__init__(in_features, out_features, bias)

    def forward(self, x, mask):
        params = self.weight * mask
        return F.linear(x, params, self.bias)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
