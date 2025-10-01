import math
import torch
from torch import nn
from torch.nn import Parameter


class CoreKernelTensorRing(nn.Module):

    def __init__(self, first_rank, m, second_rank):
        super(CoreKernelTensorRing, self).__init__()
        self.first_rank = first_rank
        self.m = m
        self.second_rank = second_rank
        self.weight = Parameter(torch.randn(first_rank, m, second_rank))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input):
        return torch.matmul(input, self.weight)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(self.
            in_features, self.out_features, self.bias is not None)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'first_rank': 4, 'm': 4, 'second_rank': 4}]
