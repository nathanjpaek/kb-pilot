import torch
import torch.nn as nn


class MinMaxNorm(nn.Module):

    def __init__(self, min, max, a=0, b=1):
        super(MinMaxNorm, self).__init__()
        self.min, self.max = min, max
        self.a, self.b = a, b

    def forward(self, x):
        return self.a + (x - self.min) * (self.b - self.a) / (self.max -
            self.min)

    def inverse(self, x):
        return self.min + (x - self.a) * (self.max - self.min) / (self.b -
            self.a)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'min': 4, 'max': 4}]
