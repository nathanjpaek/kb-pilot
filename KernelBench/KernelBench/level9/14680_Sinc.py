import torch
import torch.nn as nn


class Sinc(nn.Module):

    def forward(self, x, epsilon=1e-09):
        return torch.sin(x + epsilon) / (x + epsilon)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
