import torch
import torch.nn as nn
import torch.utils.cpp_extension


class DeNormalize(nn.Module):

    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        return x.mul(self.std).add(self.mean)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'mean': 4, 'std': 4}]
