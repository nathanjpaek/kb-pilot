import torch
import torch.nn as nn


class MyLayerNorm(nn.Module):

    def __init__(self, input_dim):
        super(MyLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(input_dim))
        if True or use_bias:
            self.beta = nn.Parameter(torch.ones(input_dim))

    def forward(self, x):
        dims = 2
        mean = x.mean(dim=dims, keepdim=True)
        x_shifted = x - mean
        var = torch.mean(x_shifted ** 2, dim=dims, keepdim=True)
        inv_std = torch.rsqrt(var + 1e-05)
        output = self.gamma * x_shifted * inv_std
        if True or use_bias:
            output += self.beta
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4}]
