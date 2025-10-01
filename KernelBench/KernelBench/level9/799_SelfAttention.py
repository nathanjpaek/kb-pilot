import torch
from torch import nn


class SelfAttention(nn.Module):

    def __init__(self, dim, heads=8):
        super(SelfAttention, self).__init__()
        self.dim, self.heads = dim, heads
        self.Q = nn.Linear(dim, dim * heads, bias=False)
        self.K = nn.Linear(dim, dim * heads, bias=False)
        self.V = nn.Linear(dim, dim * heads, bias=False)
        self.unify = nn.Linear(dim * heads, dim, bias=False)

    def forward(self, x):
        b, t, k = x.size()
        h = self.heads
        Q = self.Q(x).reshape(b, t, h, k)
        K = self.K(x).reshape(b, t, h, k)
        V = self.V(x).reshape(b, t, h, k)
        Q = Q.transpose(1, 2).reshape(b * h, t, k)
        K = K.transpose(1, 2).reshape(b * h, t, k)
        V = V.transpose(1, 2).reshape(b * h, t, k)
        Q /= k ** (1 / 4)
        K /= k ** (1 / 4)
        dot = torch.bmm(Q, K.transpose(1, 2))
        dot = torch.softmax(dot, dim=2)
        out = torch.bmm(dot, V).reshape(b, h, t, k)
        out = out.transpose(1, 2).reshape(b, t, h * k)
        return self.unify(out)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
