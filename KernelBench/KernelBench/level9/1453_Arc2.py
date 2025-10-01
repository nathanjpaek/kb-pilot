from torch.nn import Module
import torch
import numpy as np


class Arc2(Module):

    def __init__(self, num_bends):
        super(Arc2, self).__init__()
        self.num_bends = num_bends
        self.register_buffer('range', torch.arange(0, float(num_bends)))

    def a(self, t):
        return torch.cos(np.pi * t / 2)

    def b(self, t):
        return torch.sin(np.pi * t / 2)

    def forward(self, t):
        return torch.cat([self.a(t), 0.0 * (1.0 - t - self.a(t)), 0.0 * (t -
            self.b(t)), self.b(t)])


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_bends': 4}]
