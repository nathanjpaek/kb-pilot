import torch
import torch.utils.data
import torch.nn as nn


class MaxBlock(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        xm, _ = x.max(dim=1, keepdim=True)
        x = self.proj(x - xm)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4}]
