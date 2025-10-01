import torch
import torch.nn as nn


class LinearRegression(nn.Module):

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1, requires_grad=True, dtype=
            torch.float))
        self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=
            torch.float))

    def forward(self, x):
        return self.a + self.b * x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
