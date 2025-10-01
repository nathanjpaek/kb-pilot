import torch
from torch import nn


class ScalableTanh(nn.Module):

    def __init__(self, input_size):
        super(ScalableTanh, self).__init__()
        self.scale = nn.Parameter(torch.zeros(input_size), requires_grad=True)

    def forward(self, x):
        return self.scale * torch.tanh(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4}]
