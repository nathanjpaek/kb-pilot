import torch
import torch.nn as nn


class GELU(nn.Module):

    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
