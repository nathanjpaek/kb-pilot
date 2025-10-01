import torch
from torch import nn
import torch.nn.functional as F


class SelfAttentionBatch(nn.Module):

    def __init__(self, dim, da, alpha=0.2, dropout=0.5):
        super(SelfAttentionBatch, self).__init__()
        self.dim = dim
        self.da = da
        self.alpha = alpha
        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(self.dim, self.da)),
            requires_grad=True)
        self.b = nn.Parameter(torch.zeros(size=(self.da, 1)), requires_grad
            =True)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.xavier_uniform_(self.b.data, gain=1.414)

    def forward(self, h):
        e = torch.matmul(torch.tanh(torch.matmul(h, self.a)), self.b).squeeze(
            dim=1)
        attention = F.softmax(e, dim=0)
        return torch.matmul(attention, h)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'dim': 4, 'da': 4}]
