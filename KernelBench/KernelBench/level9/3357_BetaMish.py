import torch
import torch.nn as nn


class BetaMish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        beta = 1.5
        return x * torch.tanh(torch.log(torch.pow(1 + torch.exp(x), beta)))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
