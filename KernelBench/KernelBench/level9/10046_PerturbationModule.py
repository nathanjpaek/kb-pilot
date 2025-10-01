import torch
import torch.utils.data
import torch
import torch.nn as nn


class PerturbationModule(nn.Module):

    def __init__(self, T):
        super(PerturbationModule, self).__init__()
        self.T = T
        self.training = False
        self.conv_block = None

    def forward(self, x):
        if not self.training:
            x = x + self.T * torch.normal(torch.zeros_like(x), 1.0)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'T': 4}]
