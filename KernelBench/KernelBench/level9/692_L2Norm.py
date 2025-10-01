import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from torch.nn import functional as F


class L2Norm(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert x.dim(
            ) == 2, 'the input tensor of L2Norm must be the shape of [B, C]'
        return F.normalize(x, p=2, dim=-1)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
