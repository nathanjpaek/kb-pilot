import torch
import torch.nn.functional as F
from torch import nn


class Mish(nn.Module):

    def forward(self, x):
        return x * F.softplus(x).tanh()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
