import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EqualLinear(nn.Module):

    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1.0,
        activation=None):
        """

    :param in_dim:
    :param out_dim:
    :param bias:
    :param bias_init:
    :param lr_mul: 0.01
    :param activation: None: Linear; fused_leaky_relu
    """
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None
        self.activation = activation
        if self.activation is not None:
            self.act_layer = nn.LeakyReLU(0.2)
        self.scale = 1 / math.sqrt(in_dim) * lr_mul
        self.lr_mul = lr_mul
        pass

    def forward(self, input):
        out = F.linear(input, self.weight * self.scale, bias=self.bias *
            self.lr_mul)
        if self.activation:
            out = self.act_layer(out)
        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]}), activation={self.activation}'
            )


class FiLMLayerEqualFC(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer = EqualLinear(input_dim, hidden_dim)
        pass

    def forward(self, x, freq, phase_shift):
        """

    :param x: (b, num_points, d)
    :param freq: (b, d)
    :param phase_shift: (b, d)
    :return:
    """
        x = self.layer(x)
        freq = freq.unsqueeze(1).expand_as(x)
        phase_shift = phase_shift.unsqueeze(1).expand_as(x)
        out = torch.sin(freq * x + phase_shift)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'hidden_dim': 4}]
