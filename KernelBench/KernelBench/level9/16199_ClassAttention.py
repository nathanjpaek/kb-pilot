import torch
from torch import Tensor
from torch import nn


class ClassAttention(nn.Module):

    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: 'Tensor') ->Tensor:
        B, N, C = x.shape
        q = self.q(x[:, :1, :]).reshape(B, self.num_heads, 1, C // self.
            num_heads)
        q *= self.scale
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads
            ).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        cls_embed = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        cls_embed = self.proj(cls_embed)
        return cls_embed


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4, 'num_heads': 4}]
