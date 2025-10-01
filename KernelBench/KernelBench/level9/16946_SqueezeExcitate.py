import torch
from torch import nn
import torch.nn.functional as F


class SqueezeExcitate(nn.Module):

    def __init__(self, in_channels, se_size, activation=None):
        super(SqueezeExcitate, self).__init__()
        self.dim_reduce = nn.Conv2d(in_channels=in_channels, out_channels=
            se_size, kernel_size=1)
        self.dim_restore = nn.Conv2d(in_channels=se_size, out_channels=
            in_channels, kernel_size=1)
        self.activation = F.relu if activation is None else activation

    def forward(self, x):
        inp = x
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.dim_reduce(x)
        x = self.activation(x)
        x = self.dim_restore(x)
        x = torch.sigmoid(x)
        return torch.mul(inp, x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'se_size': 4}]
