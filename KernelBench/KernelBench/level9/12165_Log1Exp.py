import torch
from torch import nn


def log1exp(x):
    return torch.log(1.0 + torch.exp(x))


class Log1Exp(nn.Module):

    def forward(self, x):
        return log1exp(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
