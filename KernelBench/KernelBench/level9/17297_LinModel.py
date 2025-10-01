import torch
import torch.nn as nn
import torch.nn.functional as F


class LinModel(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(LinModel, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        out = self.linear(x)
        out = F.softmax(out, dim=-1)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4}]
