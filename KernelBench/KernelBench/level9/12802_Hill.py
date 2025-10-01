import torch
import torch.nn as nn


class Hill(nn.Module):

    def forward(self, p):
        n = 2
        return 1 / (1 + p ** n)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
