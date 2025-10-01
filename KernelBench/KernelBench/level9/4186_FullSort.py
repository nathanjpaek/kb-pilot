import torch
from torch import nn
import torch.utils.data.distributed


class FullSort(nn.Module):

    def forward(self, x):
        return torch.sort(x, 1)[0]


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
