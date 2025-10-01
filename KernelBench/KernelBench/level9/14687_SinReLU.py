import torch
import torch.nn as nn


class SinReLU(nn.Module):

    def forward(self, x):
        return torch.sin(x) + torch.relu(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
