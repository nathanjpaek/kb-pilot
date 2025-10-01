import torch
import torch.nn as nn


class CosReLU(nn.Module):

    def forward(self, x):
        return torch.cos(x) + torch.relu(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
