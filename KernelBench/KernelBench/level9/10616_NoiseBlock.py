import torch
import torch.nn as nn
import torch.jit


class NoiseBlock(nn.Module):

    def __init__(self, sigma):
        super(NoiseBlock, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        out = x + self.sigma * torch.randn_like(x)
        return out

    def set_sigma(self, x):
        self.sigma = x
        return 1


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'sigma': 4}]
