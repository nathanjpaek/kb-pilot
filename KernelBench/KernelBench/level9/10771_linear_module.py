import torch
import torch.nn as nn


class linear_module(nn.Module):
    """Module of the linear model. Inherited from nn.Module"""

    def __init__(self):
        """linear module init"""
        super(linear_module, self).__init__()
        self.a = nn.Parameter(torch.tensor(10.0))
        self.b = nn.Parameter(torch.tensor(20.0))

    def forward(self, x, y):
        """linear module forward"""
        return torch.abs(self.a * x + self.b - y)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
