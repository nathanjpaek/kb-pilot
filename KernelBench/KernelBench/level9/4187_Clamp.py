import torch
from torch import nn
import torch.utils.data


class Clamp(nn.Module):

    def __init__(self, min_out=-3, max_out=3):
        super().__init__()
        self.min_out = min_out
        self.max_out = max_out

    def forward(self, input):
        return input.clamp(self.min_out, self.max_out)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
