import math
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F


class OutlookAttention(nn.Module):

    def __init__(self, dim, num_heads, k=3, s=1, p=1):
        super().__init__()
        self.s = s
        self.k = k
        self.p = p
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.v = nn.Linear(dim, dim, bias=False)
        self.attn = nn.Linear(dim, k ** 4 * num_heads)
        self.proj = nn.Linear(dim, dim)
        self.unfold = nn.Unfold(k, padding=p, stride=s)
        self.pool = nn.AvgPool2d(s, s, ceil_mode=True)

    def forward(self, x: 'Tensor') ->Tensor:
        B, H, W, C = x.shape
        v = self.v(x).permute(0, 3, 1, 2)
        h, w = math.ceil(H / self.s), math.ceil(W / self.s)
        v = self.unfold(v).reshape(B, self.num_heads, C // self.num_heads, 
            self.k * self.k, h * w).permute(0, 1, 4, 3, 2)
        attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        attn = self.attn(attn).reshape(B, h * w, self.num_heads, self.k *
            self.k, self.k * self.k).permute(0, 2, 1, 3, 4)
        attn *= self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.k * self.
            k, h * w)
        x = F.fold(x, (H, W), self.k, padding=self.p, stride=self.s)
        x = self.proj(x.permute(0, 2, 3, 1))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4, 'num_heads': 4}]
