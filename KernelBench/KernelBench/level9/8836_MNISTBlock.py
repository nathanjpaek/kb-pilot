import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTBlock(nn.Module):

    def __init__(self, width, scaling=1.0, use_bias=True):
        super(MNISTBlock, self).__init__()
        self.scaling = scaling
        self.linear = nn.Linear(width, width, bias=use_bias)
        nn.init.xavier_normal_(self.linear.weight)
        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.linear(x)
        out = float(self.scaling) * F.relu(out)
        out += self.shortcut(x)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'width': 4}]
