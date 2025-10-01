import torch
import torch.nn as nn
from torch.nn import Parameter


class HighWay(torch.nn.Module):

    def __init__(self, f_in, f_out, bias=True):
        super(HighWay, self).__init__()
        self.w = Parameter(torch.Tensor(f_in, f_out))
        nn.init.xavier_uniform_(self.w)
        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)

    def forward(self, in_1, in_2):
        t = torch.mm(in_1, self.w)
        if self.bias is not None:
            t = t + self.bias
        gate = torch.sigmoid(t)
        return gate * in_2 + (1.0 - gate) * in_1


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'f_in': 4, 'f_out': 4}]
