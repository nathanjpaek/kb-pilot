import torch
import torch.nn as nn
import torch.nn.parallel


class WShift(nn.Module):

    def __init__(self, style_dim):
        super().__init__()
        self.w_shift = nn.Parameter(torch.zeros(1, style_dim))

    def forward(self, input):
        out = input + self.w_shift
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'style_dim': 4}]
