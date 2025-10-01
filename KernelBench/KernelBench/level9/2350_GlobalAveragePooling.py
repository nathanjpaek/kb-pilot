import torch
import torch.nn as nn


class GlobalAveragePooling(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mean([2, 3])


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
