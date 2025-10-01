import torch
import torch.nn as nn


class InverseSqrt(nn.Module):

    def forward(self, x, alpha=1.0):
        return x / torch.sqrt(1.0 + alpha * x * x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
