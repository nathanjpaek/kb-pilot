import torch
import torch as th
import torch.nn as nn


class SiLU(nn.Module):

    def forward(self, x):
        return x * th.sigmoid(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
