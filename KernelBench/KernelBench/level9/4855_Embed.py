import torch
import torch.nn.functional as F
import torch.nn as nn
import torch._utils
from itertools import product as product
import torch.utils.data.distributed


class Embed(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(Embed, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = F.normalize(x, p=2, dim=1)
        return x


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4}]
