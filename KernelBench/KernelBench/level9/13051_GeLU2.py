import torch
import torch.nn as nn


class GeLU2(nn.Module):

    def forward(self, x):
        return (1.702 * x).sigmoid() * x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
