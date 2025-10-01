import torch
from torch import nn


class GramMatrix(nn.Module):

    def forward(self, x):
        b, c, h, w = x.shape
        F = x.view(-1, c, b * w)
        G = torch.bmm(F, F.transpose(1, 2)) / (h * w)
        return G


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
