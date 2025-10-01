import torch
import torch.nn as nn


def to_2tuple(value):
    return value, value


class CyclicShift(nn.Module):

    def __init__(self, displacement):
        super().__init__()
        if isinstance(displacement, int):
            self.displacement = to_2tuple(displacement)
        else:
            self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement[0], self.
            displacement[0]), dims=(1, 2))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'displacement': 4}]
