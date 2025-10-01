import torch
from torch import nn


class NoiseInjection(nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image):
        noise = torch.randn_like(image[:, 0:1, :, :])
        return image + self.weight * noise * 0.9


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
