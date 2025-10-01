from torch.nn import Module
import torch


class Tanh(Module):
    """Rectified Tanh, since we predict betwee 0 and 1"""

    def __init__(self):
        super().__init__()
        self.params = []

    def forward(self, x):
        self.x = x
        return 0.5 * (1 + x.tanh())

    def backward(self, d_dx):
        return 0.5 * d_dx * (1 - torch.tanh(self.x) ** 2)

    def param(self):
        return self.params


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
