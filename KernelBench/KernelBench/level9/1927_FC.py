import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F


class FC(nn.Module):

    def __init__(self, in_channels, out_channels, gain=2 ** 0.5, use_wscale
        =False, lrmul=1.0, bias=True):
        super(FC, self).__init__()
        he_std = gain * in_channels ** -0.5
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul
        self.weight = torch.nn.Parameter(torch.randn(out_channels,
            in_channels) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
            self.b_lrmul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            out = F.linear(x, self.weight * self.w_lrmul, self.bias * self.
                b_lrmul)
        else:
            out = F.linear(x, self.weight * self.w_lrmul)
        out = F.leaky_relu(out, 0.2, inplace=True)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
