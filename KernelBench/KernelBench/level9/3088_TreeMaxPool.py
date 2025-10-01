import torch
from torch import nn
import torch.utils.data


class TreeMaxPool(nn.Module):

    def forward(self, trees):
        return trees[0].max(dim=2).values


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
