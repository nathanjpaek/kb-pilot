import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import *


class SelfAttentionLayer(nn.Module):

    def __init__(self, dim, da, alpha=0.2, dropout=0.5):
        super(SelfAttentionLayer, self).__init__()
        self.dim = dim
        self.da = da
        self.alpha = alpha
        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(self.dim, self.da)))
        self.b = nn.Parameter(torch.zeros(size=(self.da, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.xavier_uniform_(self.b.data, gain=1.414)

    def forward(self, h):
        h.shape[0]
        assert self.dim == h.shape[1]
        e = torch.matmul(torch.tanh(torch.matmul(h, self.a)), self.b).squeeze(
            dim=1)
        attention = F.softmax(e)
        return torch.matmul(attention, h)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'dim': 4, 'da': 4}]
