import torch
import torch.nn as nn


class Triangle(nn.Module):

    def forward(self, x):
        return x.abs()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
