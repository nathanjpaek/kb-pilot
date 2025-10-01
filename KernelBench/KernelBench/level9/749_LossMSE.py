from torch.nn import Module
import torch


class LossMSE(Module):
    """implementation of the Mean-Squared Error Loss"""

    def __init__(self):
        super().__init__()
        self.params = []

    def forward(self, y, t):
        self.y = y
        self.t = t
        return torch.dist(y, t, p=2)

    def backward(self):
        return 2 * (self.y - self.t)

    def param(self):
        return self.params


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
