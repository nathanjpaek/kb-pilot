import torch
import torch.nn as nn


class Swish(nn.Module):

    def forward(self, x):
        return x.mul_(torch.sigmoid(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
