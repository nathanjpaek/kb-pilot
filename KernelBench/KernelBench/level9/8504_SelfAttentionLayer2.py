import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import *


class SelfAttentionLayer2(nn.Module):

    def __init__(self, dim, da):
        super(SelfAttentionLayer2, self).__init__()
        self.dim = dim
        self.Wq = nn.Parameter(torch.zeros(self.dim, self.dim))
        self.Wk = nn.Parameter(torch.zeros(self.dim, self.dim))
        nn.init.xavier_uniform_(self.Wq.data, gain=1.414)
        nn.init.xavier_uniform_(self.Wk.data, gain=1.414)

    def forward(self, h):
        h.shape[0]
        assert self.dim == h.shape[1]
        q = torch.matmul(h, self.Wq)
        k = torch.matmul(h, self.Wk)
        e = torch.matmul(q, k.t()) / math.sqrt(self.dim)
        attention = F.softmax(e, dim=1)
        attention = attention.mean(dim=0)
        x = torch.matmul(attention, h)
        return x


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'dim': 4, 'da': 4}]
