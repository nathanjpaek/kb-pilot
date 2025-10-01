import math
import torch


class SineODE(torch.nn.Module):

    def forward(self, t, y):
        return 2 * y / t + t ** 4 * torch.sin(2 * t) - t ** 2 + 4 * t ** 3

    def y_exact(self, t):
        return -0.5 * t ** 4 * torch.cos(2 * t) + 0.5 * t ** 3 * torch.sin(
            2 * t) + 0.25 * t ** 2 * torch.cos(2 * t) - t ** 3 + 2 * t ** 4 + (
            math.pi - 0.25) * t ** 2


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
