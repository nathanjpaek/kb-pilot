import torch
import torch.nn as nn


class NormLayer(nn.Module):

    def __init__(self, mean, std, n=None, eps=1e-08) ->None:
        super().__init__()
        self.mean = mean
        self.std = std
        self.eps = eps

    def forward(self, x):
        return (x - self.mean) / (self.std + self.eps)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'mean': 4, 'std': 4}]
