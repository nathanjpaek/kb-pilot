import math
import torch
from torch import nn
from torch.nn import functional as F


class EqualLinearWithLeakyRelu(nn.Module):
    """Add this class for onnx -- data driven flow is difficult tracing."""

    def __init__(self, in_dim, out_dim, lr_mul=0.01):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.scale = 1 / math.sqrt(in_dim) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        out = F.linear(input, self.weight * self.scale)
        shape = 1, self.bias.shape[0]
        new_bias = self.bias * self.lr_mul
        return 1.414 * F.leaky_relu(out + new_bias.view(shape),
            negative_slope=0.2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4}]
