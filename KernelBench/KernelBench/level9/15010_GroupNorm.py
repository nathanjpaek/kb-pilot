import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data


class GroupNorm(nn.Module):

    def __init__(self, num_groups, embed_dim, eps=1e-05, affine=True):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, embed_dim, eps, affine)

    def forward(self, x):
        B, T, C = x.shape
        x = x.view(B * T, C)
        x = self.gn(x)
        x = x.view(B, T, C)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'num_groups': 1, 'embed_dim': 4}]
