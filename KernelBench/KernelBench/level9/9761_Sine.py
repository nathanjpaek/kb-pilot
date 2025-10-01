import torch
import torch.nn as nn


class Sine(nn.Module):

    def __init__(self, w0: 'float'=30.0):
        super(Sine, self).__init__()
        self.w0 = w0

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return torch.sin(self.w0 * x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
