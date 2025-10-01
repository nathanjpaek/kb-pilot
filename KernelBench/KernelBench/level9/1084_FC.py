import torch
import torch.nn as nn
import torch.nn.functional as F


class FC(nn.Module):

    def __init__(self, in_channels, out_channels, use_bias=False,
        activation='LR', gain=2 ** 0.5):
        super(FC, self).__init__()
        self.he_std = in_channels * -0.5 * gain
        self.weight = torch.nn.Parameter(torch.randn(out_channels,
            in_channels) * self.he_std)
        if use_bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        if activation == 'LR':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation is None:
            self.activation = None
        else:
            assert 0, " STGAN's FC reruires LR or Sigmoid, not{}".format(
                activation)

    def forward(self, x):
        if self.bias is not None:
            out = F.linear(x, self.weight, self.bias)
        else:
            out = F.linear(x, self.weight)
        if self.activation:
            out = self.activation(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
