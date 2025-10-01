import torch
import torch.nn as nn


class DummyLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.ones(1, dtype=torch.float))

    def forward(self, x):
        return x + self.dummy - self.dummy


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
