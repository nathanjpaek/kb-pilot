import torch
import torch.nn as nn
import torch.nn.functional as F


class last_fc(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(last_fc, self).__init__(in_features, out_features, bias)
        self.layer_type = 'LFC'
        self.transform = None

    def forward(self, x):
        restore_w = self.weight
        max = restore_w.data.max()
        weight_q = restore_w.div(max).mul(127).round().div(127).mul(max)
        weight_q = (weight_q - restore_w).detach() + restore_w
        return F.linear(x, weight_q, self.bias)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
