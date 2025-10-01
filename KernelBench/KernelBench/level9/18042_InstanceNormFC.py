import torch
from torch import nn


class InstanceNormFC(nn.Module):

    def __init__(self, _unused=0, affine=True):
        super().__init__()
        self.norm = nn.InstanceNorm1d(1, affine=affine)

    def forward(self, x):
        return self.norm(x.unsqueeze(1)).squeeze(1)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
