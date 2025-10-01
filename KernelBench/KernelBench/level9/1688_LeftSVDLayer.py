import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class LeftSVDLayer(nn.Module):

    def __init__(self, ih, oh, dropout=None, bias=True):
        super().__init__()
        self.weight = Parameter(torch.Tensor(oh, ih))
        self.dropout = dropout
        if bias:
            self.bias = Parameter(torch.Tensor(oh, 1))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fin, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fin / 2.0)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        y = self.weight.matmul(x)
        if self.bias is not None:
            y = y + self.bias
        if self.dropout is not None:
            y = F.dropout(y, p=self.dropout)
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'ih': 4, 'oh': 4}]
