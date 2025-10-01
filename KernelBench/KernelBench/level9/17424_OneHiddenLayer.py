import torch
import torch as tr
import torch.nn as nn


class quadexp(nn.Module):

    def __init__(self, sigma=2.0):
        super(quadexp, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        return tr.exp(-x ** 2 / self.sigma ** 2)


class OneHiddenLayer(nn.Module):

    def __init__(self, d_int, H, d_out, non_linearity=quadexp(), bias=False):
        super(OneHiddenLayer, self).__init__()
        self.linear1 = tr.nn.Linear(d_int, H, bias=bias)
        self.linear2 = tr.nn.Linear(H, d_out, bias=bias)
        self.non_linearity = non_linearity
        self.d_int = d_int
        self.d_out = d_out

    def weights_init(self, center, std):
        self.linear1.weights_init(center, std)
        self.linear2.weights_init(center, std)

    def forward(self, x):
        h1_relu = self.linear1(x).clamp(min=0)
        h2_relu = self.linear2(h1_relu)
        h2_relu = self.non_linearity(h2_relu)
        return h2_relu


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_int': 4, 'H': 4, 'd_out': 4}]
