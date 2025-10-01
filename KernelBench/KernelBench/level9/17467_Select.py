import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.parameter import Parameter


class Select(nn.Module):

    def __init__(self, c):
        super(Select, self).__init__()
        self.weight = Parameter(torch.ones(c, requires_grad=False))

    def forward(self, input):
        """
        input_tensor: (N,C,H,W)
        """
        weight = self.weight[None, :, None, None]
        out = input * weight
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'c': 4}]
