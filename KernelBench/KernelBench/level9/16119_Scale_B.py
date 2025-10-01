import torch
import torch.nn as nn


class Scale_B(nn.Module):
    """
    Learned per-channel scale factor, used to scale the noise
    """

    def __init__(self, n_channel):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros((1, n_channel, 1, 1)))

    def forward(self, noise):
        result = noise * self.weight
        return result


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_channel': 4}]
