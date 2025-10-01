import torch
import torch.nn as nn
from string import ascii_lowercase
import torch.optim


class Collapse(nn.Module):

    def __init__(self, size):
        super(Collapse, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(size), requires_grad=True)
        self.weight.data.zero_()
        self.p_avg_l = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.p_max_l = nn.AdaptiveMaxPool2d(output_size=(1, 1))

    def forward(self, x):
        return self.collapse(x)

    def collapse(self, inputs):
        p_avg = self.p_avg_l(inputs)
        p_max = self.p_max_l(inputs)
        factor = torch.sigmoid(self.weight)
        eqn = 'ay{0},y->ay{0}'.format(ascii_lowercase[1:3])
        return torch.einsum(eqn, [p_avg, factor]) + torch.einsum(eqn, [
            p_max, torch.sub(1.0, factor)])


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'size': 4}]
