import torch
import torch.nn.functional as F
import torch.nn as nn


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
