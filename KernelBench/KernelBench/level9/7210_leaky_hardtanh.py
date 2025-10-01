import torch
import torch.nn as nn


class leaky_hardtanh(nn.Module):

    def __init__(self, min=-1, max=1, slope=0.01):
        super(leaky_hardtanh, self).__init__()
        self.min = min
        self.max = max
        self.slope = slope

    def forward(self, x):
        x = torch.where(x < self.min, self.min + x * self.slope, x)
        x = torch.where(x > self.max, self.max + x * self.slope, x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
