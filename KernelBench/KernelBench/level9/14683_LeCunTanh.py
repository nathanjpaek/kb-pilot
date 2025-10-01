import torch
import torch.nn as nn


class LeCunTanh(nn.Module):

    def forward(self, x):
        return 1.7159 * torch.tanh(2.0 / 3 * x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
