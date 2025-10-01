import torch
import torch.nn as nn


class ActivationFunction(nn.Module):

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.config = {'name': self.name}


class LeakyReLU(ActivationFunction):

    def __init__(self, alpha=0.1):
        super().__init__()
        self.config['alpha'] = alpha

    def forward(self, x):
        return torch.where(x > 0, x, self.config['alpha'] * x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
