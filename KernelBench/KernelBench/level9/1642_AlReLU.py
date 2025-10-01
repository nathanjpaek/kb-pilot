import torch
from torch import nn


class AlReLU(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return self.alrelu(input)

    def alrelu(self, x):
        alpha = 0.01
        return torch.maximum(torch.abs(alpha * x), x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
