import torch
import numpy as np
import torch.nn as nn


class FAdd(nn.Module):

    def __init__(self):
        super(FAdd, self).__init__()

    def forward(self, x, y):
        x = x + y + np.float32(0.1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
