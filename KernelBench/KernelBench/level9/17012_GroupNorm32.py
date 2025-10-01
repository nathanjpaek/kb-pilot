import torch
import torch.nn.functional as F
import torch.nn as nn


class GroupNorm32(nn.GroupNorm):

    def __init__(self, num_groups, num_channels, swish, eps=1e-05):
        super().__init__(num_groups=num_groups, num_channels=num_channels,
            eps=eps)
        self.swish = swish

    def forward(self, x):
        y = super().forward(x.float())
        if self.swish == 1.0:
            y = F.silu(y)
        elif self.swish:
            y = y * F.sigmoid(y * float(self.swish))
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_groups': 1, 'num_channels': 4, 'swish': 4}]
