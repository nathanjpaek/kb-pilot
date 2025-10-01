import torch
import torch.nn as nn


class SimpleDropoutOptimizer(nn.Module):

    def __init__(self, p):
        super().__init__()
        if p is not None:
            self.dropout = nn.Dropout(p=p)
        else:
            self.dropout = None

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'p': 0.5}]
