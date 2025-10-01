import torch
import torch.nn as nn


class CPULayerNorm(nn.Module):

    def __init__(self, features, eps=1e-06):
        super().__init__()
        self.features = features
        self.eps = eps

    def forward(self, x, gamma, beta):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return gamma * ((x - mean) / (std + self.eps)) + beta


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'features': 4}]
