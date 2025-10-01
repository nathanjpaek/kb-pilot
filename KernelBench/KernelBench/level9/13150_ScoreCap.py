import torch
from torch import nn
import torch.nn
import torch.optim


class ScoreCap(nn.Module):

    def __init__(self, cap: 'float'):
        super().__init__()
        self.cap = cap

    def forward(self, input):
        return torch.clip(input, max=self.cap)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'cap': 4}]
