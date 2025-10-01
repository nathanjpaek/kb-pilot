import torch
from torch import nn
from torch import stack


class Normalization(nn.Module):

    def __init__(self, S_low, S_up, a_low, a_up, **kwargs):
        super(Normalization, self).__init__(**kwargs)
        self.low_bound_S = S_low
        self.upper_bound_S = S_up
        self.low_bound_a = a_low
        self.upper_bound_a = a_up

    def forward(self, x):
        s = x[:, 0]
        a = x[:, 1]
        s = (s - self.low_bound_S) / (self.upper_bound_S - self.low_bound_S)
        a = (a - self.low_bound_a) / (self.upper_bound_a - self.low_bound_a)
        return stack((s, a)).T


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'S_low': 4, 'S_up': 4, 'a_low': 4, 'a_up': 4}]
