import torch
import torch.nn as nn


class BentIdentity(nn.Module):

    def forward(self, x, alpha=1.0):
        return x + (torch.sqrt(1.0 + x * x) - 1.0) / 2.0


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
