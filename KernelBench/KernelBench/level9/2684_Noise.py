import torch
from torch import nn


def exists(val):
    return val is not None


class Noise(nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x, noise=None):
        b, _, h, w, device = *x.shape, x.device
        if not exists(noise):
            noise = torch.randn(b, 1, h, w, device=device)
        return x + self.weight * noise


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
