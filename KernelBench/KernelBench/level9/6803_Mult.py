import torch
import torch.utils.data
import torch
from torch import nn


class Mult(nn.Module):

    def __init__(self, nc):
        super(Mult, self).__init__()
        self.register_parameter(name='exp', param=torch.nn.Parameter(torch.
            diag(torch.ones(nc)).unsqueeze(-1).unsqueeze(-1)))
        """self.register_parameter(name='weight',
                                param=torch.nn.Parameter(torch.ones(nc).unsqueeze(-1).unsqueeze(-1)))
                                """
        self.register_parameter(name='bias', param=torch.nn.Parameter(torch
            .zeros(nc).unsqueeze(-1).unsqueeze(-1)))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(x) + 0.1
        return x.unsqueeze(-3).pow(self.exp).prod(1) + self.bias


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nc': 4}]
