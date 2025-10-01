import torch
import torch.nn as nn


class SelfExpression(nn.Module):

    def __init__(self, n):
        super(SelfExpression, self).__init__()
        self.Coefficient = nn.Parameter(1e-08 * torch.ones(n, n, dtype=
            torch.float32), requires_grad=True)

    def forward(self, x):
        y = torch.matmul(self.Coefficient, x)
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n': 4}]
