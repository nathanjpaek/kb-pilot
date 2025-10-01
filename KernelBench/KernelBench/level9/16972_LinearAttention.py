import torch
from torch import nn


class LinearAttention(nn.Module):

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, self.hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(self.hidden_dim, dim, 1)

    def forward(self, x):
        b, _, h, w = x.shape
        qkv = self.to_qkv(x)
        qkv = qkv.reshape(b, 3, self.heads, self.dim_head, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        k = k.softmax(dim=-1)
        context = torch.matmul(k, v.permute(0, 1, 3, 2))
        out = torch.matmul(context.permute(0, 1, 3, 2), q)
        out = out.reshape(b, -1, h, w)
        return self.to_out(out)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
