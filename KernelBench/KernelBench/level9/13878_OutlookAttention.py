import math
import torch
from torch import nn
from torch.nn import functional as F


class OutlookAttention(nn.Module):

    def __init__(self, dim, num_heads=1, kernel_size=3, padding=1, stride=1,
        qkv_bias=False, attn_drop=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.scale = self.head_dim ** -0.5
        self.v_pj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn = nn.Linear(dim, kernel_size ** 4 * num_heads)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(attn_drop)
        self.unflod = nn.Unfold(kernel_size, padding, stride)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride,
            ceil_mode=True)

    def forward(self, x):
        B, H, W, C = x.shape
        v = self.v_pj(x).permute(0, 3, 1, 2)
        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)
        v = self.unflod(v).reshape(B, self.num_heads, self.head_dim, self.
            kernel_size * self.kernel_size, h * w).permute(0, 1, 4, 3, 2)
        attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        attn = self.attn(attn).reshape(B, h * w, self.num_heads, self.
            kernel_size * self.kernel_size, self.kernel_size * self.kernel_size
            ).permute(0, 2, 1, 3, 4)
        attn = self.scale * attn
        attn = attn.softmax(-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.
            kernel_size * self.kernel_size, h * w)
        out = F.fold(out, output_size=(H, W), kernel_size=self.kernel_size,
            padding=self.padding, stride=self.stride)
        out = self.proj(out.permute(0, 2, 3, 1))
        out = self.proj_drop(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
