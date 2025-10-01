import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class LinearCaps(nn.Module):

    def __init__(self, in_features, num_C, num_D, bias=False, eps=0.0001):
        super(LinearCaps, self).__init__()
        self.in_features = in_features
        self.num_C = num_C
        self.num_D = num_D
        self.eps = eps
        self.weight = Parameter(torch.Tensor(num_C * num_D, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(num_C * num_D))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        scalar = torch.sqrt(torch.sum(self.weight * self.weight, dim=1))
        scalar = torch.reciprocal(scalar + self.eps)
        scalar = torch.unsqueeze(scalar, dim=1)
        output = F.linear(x, scalar * self.weight, self.bias)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'num_C': 4, 'num_D': 4}]
