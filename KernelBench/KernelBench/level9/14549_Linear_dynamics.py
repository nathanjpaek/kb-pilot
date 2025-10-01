import torch
import torch.utils.data
from torch import nn


class Linear_dynamics(nn.Module):

    def __init__(self, device='cpu'):
        super(Linear_dynamics, self).__init__()
        self.time = nn.Parameter(torch.ones(1) * 0.7)
        self.device = device
        self

    def forward(self, x, v):
        return x + v * self.time


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
