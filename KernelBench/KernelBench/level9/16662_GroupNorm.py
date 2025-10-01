import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch._utils
import torch.optim


def num_groups(group_size, channels):
    if not group_size:
        return 1
    else:
        assert channels % group_size == 0
        return channels // group_size


class GroupNorm(nn.GroupNorm):

    def __init__(self, num_channels, num_groups, eps=1e-05, affine=True):
        super().__init__(num_groups, num_channels, eps=eps, affine=affine)

    def forward(self, x):
        return F.group_norm(x, self.num_groups, self.weight, self.bias,
            self.eps)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_channels': 4, 'num_groups': 1}]
