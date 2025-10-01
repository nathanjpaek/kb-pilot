import torch
import torch.nn as nn


class StandardNorm(nn.Module):

    def __init__(self, mean, std):
        super(StandardNorm, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'mean': 4, 'std': 4}]
