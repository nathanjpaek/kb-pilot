import torch
import torch.nn as nn


class SPU(nn.Module):

    def forward(self, x):
        return torch.where(x > 0, x ** 2 - 0.5, torch.sigmoid(-x) - 1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
