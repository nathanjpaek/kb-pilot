import torch
import torch.utils.data
import torch
import torch.nn as nn


class NoiseInjection(nn.Module):

    def __init__(self, channel):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))

    def forward(self, image, noise):
        return image + self.weight * noise.unsqueeze(2).unsqueeze(3)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channel': 4}]
