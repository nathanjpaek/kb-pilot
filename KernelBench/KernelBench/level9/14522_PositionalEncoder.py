import torch
from torch import nn


class PositionalEncoder(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, xyz):
        xyz1 = xyz.unsqueeze(1)
        xyz2 = xyz.unsqueeze(0)
        pairwise_dist = xyz1 - xyz2
        return pairwise_dist


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4}]
