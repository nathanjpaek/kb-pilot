import torch
from torch import nn


class PositionalEncodingGenerator(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, 3, padding=1, bias=False, groups=dim)

    def forward(self, input):
        out = input.permute(0, 3, 1, 2)
        out = self.proj(out) + out
        out = out.permute(0, 2, 3, 1)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
