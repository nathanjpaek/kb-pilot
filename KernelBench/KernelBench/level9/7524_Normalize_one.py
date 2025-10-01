import torch
from torch import nn


class Normalize_one(nn.Module):

    def __init__(self, mean, std):
        super(Normalize_one, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, input):
        x = input.clone()
        x = (x - self.mean) / self.std
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'mean': 4, 'std': 4}]
