import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.functional as F


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 *
        torch.pow(x, 3))))


class DWConv(nn.Module):
    """
    Depthwise separable 1d convolution
    """

    def __init__(self, nin, nout, kernel_size, bias=True, act='relu'):
        super(DWConv, self).__init__()
        self.depthwise = nn.Conv1d(nin, nin, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=nin, bias=bias)
        self.pointwise = nn.Conv1d(nin, nout, kernel_size=1, bias=bias)
        self.act = act

    def forward(self, x):
        out = self.depthwise(x.permute(0, 2, 1))
        out = self.pointwise(out)
        out = out.permute(0, 2, 1)
        if self.act == 'relu':
            out = F.relu(out)
        elif self.act == 'gelu':
            out = gelu(out)
        return out


class Resizer(nn.Module):

    def __init__(self, input_size, output_size, kernel_size, drop_prob=0,
        bias=False, act=None):
        super(Resizer, self).__init__()
        self.conv = DWConv(input_size, output_size, kernel_size, bias=bias,
            act=act)
        self.drop = nn.Dropout(drop_prob)

    def forward(self, x):
        out = self.conv(x)
        return self.drop(out)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'output_size': 4, 'kernel_size': 4}]
