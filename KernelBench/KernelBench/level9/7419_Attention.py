import torch
from torch import nn


class Attention(nn.Module):

    def __init__(self, heads, dim, hidden_dim):
        super().__init__()
        self.dim = dim
        self.hdim = hidden_dim
        self.heads = heads
        self.to_q = nn.Linear(dim, hidden_dim * heads)
        self.to_k = nn.Linear(dim, hidden_dim * heads)
        self.to_v = nn.Linear(dim, hidden_dim * heads)
        self.project = nn.Linear(heads * hidden_dim, dim)

    def forward(self, x):
        B, T, _ = x.shape
        Q = self.to_q(x).view(B, self.heads, T, self.hdim)
        K = self.to_k(x).view(B, self.heads, T, self.hdim)
        V = self.to_v(x).view(B, self.heads, T, self.hdim)
        weights = torch.softmax(Q.permute(0, 1, 3, 2) @ K / self.hdim ** 
            0.5, dim=-1)
        out = weights @ V.permute(0, 1, 3, 2)
        out = self.project(out.view(B, T, self.heads * self.hdim))
        return out


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'heads': 4, 'dim': 4, 'hidden_dim': 4}]
