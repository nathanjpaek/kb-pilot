import torch
import torch.nn as nn
import torch.cuda
import torch.onnx.utils
import torch.random
import torch.cuda.random
import torch.utils.cpp_extension


class BertLayerNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        do_convert = self.weight.dtype != x.dtype
        if do_convert:
            x = x
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        x = self.weight * x + self.bias
        if do_convert:
            x = x
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
