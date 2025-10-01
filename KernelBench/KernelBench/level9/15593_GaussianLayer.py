import torch
import torch.nn as nn


class GaussianLayer(nn.Module):

    def __init__(self, std, device):
        super().__init__()
        self.std = std
        self.device = device

    def forward(self, x):
        return x + self.std * torch.randn_like(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'std': 4, 'device': 0}]
