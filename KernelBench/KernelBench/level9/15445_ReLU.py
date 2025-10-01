import torch
import torch.nn as nn


class ActivationFunction(nn.Module):

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.config = {'name': self.name}


class ReLU(ActivationFunction):

    def forward(self, x):
        return x * (x > 0).float()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
