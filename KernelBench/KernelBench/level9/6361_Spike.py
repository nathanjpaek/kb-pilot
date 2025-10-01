import torch
import torch.nn as nn


class Spike(nn.Module):

    def __init__(self, center=1, width=1):
        super(Spike, self).__init__()
        self.c = center
        self.w = width
        self.alpha = torch.nn.Parameter(torch.ones(1))
        self.beta = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.alpha * x + self.beta * (torch.min(torch.max(x - (self.
            c - self.w), torch.zeros_like(x)), torch.max(-x + (self.c +
            self.w), torch.zeros_like(x))) - 2 * torch.min(torch.max(x - (
            self.c - self.w + 1), torch.zeros_like(x)), torch.max(-x + (
            self.c + self.w + 1), torch.zeros_like(x))))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
