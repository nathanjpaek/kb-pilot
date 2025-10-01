import torch
from torch.nn import functional as F
from torch import nn


class ReluSquared(nn.Module):

    def forward(self, x):
        return F.relu(x) ** 2


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
