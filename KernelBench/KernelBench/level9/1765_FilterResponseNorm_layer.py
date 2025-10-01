import torch
import torch.nn as nn


class FilterResponseNorm_layer(nn.Module):

    def __init__(self, num_filters, eps=1e-06):
        super(FilterResponseNorm_layer, self).__init__()
        self.eps = eps
        par_shape = 1, num_filters, 1, 1
        self.tau = torch.nn.Parameter(torch.zeros(par_shape))
        self.beta = torch.nn.Parameter(torch.zeros(par_shape))
        self.gamma = torch.nn.Parameter(torch.ones(par_shape))

    def forward(self, x):
        nu2 = torch.mean(torch.square(x), dim=[2, 3], keepdim=True)
        x = x * 1 / torch.sqrt(nu2 + self.eps)
        y = self.gamma * x + self.beta
        z = torch.max(y, self.tau)
        return z


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_filters': 4}]
