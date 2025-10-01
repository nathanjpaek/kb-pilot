import torch
import torch.nn as nn


class CustomReLU(nn.Module):

    def __init__(self, max_z=6.0):
        super(CustomReLU, self).__init__()
        self.max_z = max_z

    def forward(self, x):
        return torch.clamp(x, min=0, max=self.max_z)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
