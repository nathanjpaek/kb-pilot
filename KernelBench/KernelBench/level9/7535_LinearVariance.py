import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter


class ModuleWrapper(nn.Module):
    """Wrapper for nn.Module with support for arbitrary flags and a universal forward pass"""

    def __init__(self):
        super(ModuleWrapper, self).__init__()

    def set_flag(self, flag_name, value):
        setattr(self, flag_name, value)
        for m in self.children():
            if hasattr(m, 'set_flag'):
                m.set_flag(flag_name, value)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


class LinearVariance(ModuleWrapper):

    def __init__(self, in_features, out_features, bias=True):
        super(LinearVariance, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.sigma.size(1))
        self.sigma.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x):
        lrt_mean = self.bias
        lrt_std = torch.sqrt_(1e-16 + F.linear(x * x, self.sigma * self.sigma))
        if self.training:
            eps = Variable(lrt_std.data.new(lrt_std.size()).normal_())
        else:
            eps = 0.0
        return lrt_mean + eps * lrt_std


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
