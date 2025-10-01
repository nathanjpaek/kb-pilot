import torch
import torch.nn as nn


class DWConv(nn.Module):

    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


def get_inputs():
    return [torch.rand([4, 768, 64, 64])]


def get_init_inputs():
    return [[], {}]
