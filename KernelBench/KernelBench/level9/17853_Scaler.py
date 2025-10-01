import torch
from torch import nn


class Scaler(nn.Module):

    def __init__(self, alpha=16.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, input):
        return self.alpha * input


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
