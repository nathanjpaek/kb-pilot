import torch
import torch.nn as nn
import torch.fx


class TestModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Module()
        self.b = torch.nn.Module()
        self.a.weights = torch.nn.Parameter(torch.randn(1, 2))
        self.b.weights = torch.nn.Parameter(torch.randn(1))

    def forward(self, x):
        return x + self.a.weights + self.b.weights


def get_inputs():
    return [torch.rand([4, 4, 4, 2])]


def get_init_inputs():
    return [[], {}]
