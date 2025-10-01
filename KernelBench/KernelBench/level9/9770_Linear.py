import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import Parameter
import torch.cuda
import torch.distributed


def quantize_weights(W, numbits=8):
    W = W.clamp(-2 ** (numbits - 1), 2 ** (numbits - 1))
    W = W.mul(2 ** (numbits - 1)).round().div(2 ** (numbits - 1))
    return W


class quant_weights(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        return quantize_weights(x)

    @staticmethod
    def backward(ctx, g):
        return g


class Linear(nn.modules.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.W_LR_scale = np.float32(1.0 / np.sqrt(1.5 / (in_features +
            out_features)))
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward_t(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(self.
            in_features, self.out_features, self.bias is not None)

    def forward(self, input):
        Wr = self.weight.data
        self.Wb = quant_weights.apply(self.weight.data)
        self.input_b = quant_weights.apply(input)
        self.weight.data = self.Wb
        rvalue = self.forward_t(self.input_b)
        self.weight.data = Wr
        return rvalue

    def return_W_scale(self):
        return self.W_LR_scale


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
