import torch
import torch.nn.functional as F
import torch.nn as nn


class Mish(nn.Module):

    def forward(self, x):
        return x.mul_(F.softplus(x).tanh())


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
