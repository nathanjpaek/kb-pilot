import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedLinear(nn.Linear):

    def __init__(self, in_features, out_features, share_weight=False):
        super(SharedLinear, self).__init__(in_features, out_features, bias=True
            )
        if share_weight:
            self.weight = nn.Parameter(torch.Tensor(1, in_features))
        self.reset_parameters()

    def forward(self, x):
        return F.linear(x, self.weight) + self.bias


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
