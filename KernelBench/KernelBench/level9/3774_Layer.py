import torch
import torch.nn.functional as F
import torch.nn as nn


class Layer(nn.Module):

    def __init__(self, input_dim, output_dim, p, name=None):
        super(Layer, self).__init__()
        self.name = name
        self.register_parameter(name='w', param=nn.Parameter(torch.empty(1,
            input_dim, output_dim), requires_grad=True))
        self.register_parameter(name='b', param=nn.Parameter(torch.full([1,
            1, output_dim], 0.1), requires_grad=True))
        self.p = p
        with torch.no_grad():
            nn.init.trunc_normal_(self.w, mean=0, std=0.1, a=-0.2, b=0.2)

    def forward(self, x):
        W = torch.tile(self.w, [self.p, 1, 1])
        B = torch.tile(self.b, [self.p, 1, 1])
        y = torch.matmul(x, W) + B
        return F.relu(y)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4, 'p': 4}]
