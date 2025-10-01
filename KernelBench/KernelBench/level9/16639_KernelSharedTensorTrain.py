import torch
from torch import nn
from torch.nn import Parameter


class KernelSharedTensorTrain(nn.Module):

    def __init__(self, first_rank, m, second_rank, init_value):
        super(KernelSharedTensorTrain, self).__init__()
        self.first_rank = first_rank
        self.m = m
        self.second_rank = second_rank
        self.weight = Parameter(torch.randn(first_rank, m, second_rank))
        self.init_value = init_value
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, gain=self.init_value)

    def forward(self, input, state):
        x = torch.einsum('bj,bi->bji', [input, state])
        x = torch.einsum('ijk,bji->bk', [self.weight, x])
        return x

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(self.
            in_features, self.out_features, self.bias is not None)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'first_rank': 4, 'm': 4, 'second_rank': 4, 'init_value': 4}]
