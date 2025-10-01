import math
import torch
from torch import nn
import torch.utils.data.dataloader
import torch.utils.data
import torch.onnx
import torch.backends.cudnn


class Attention(nn.Module):

    def __init__(self, dim_q, dim_kv, num_heads=4, qkv_bias=False, stride=1):
        super().__init__()
        self.dim = dim_q
        self.num_heads = num_heads
        head_dim = dim_q // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim_q, dim_q, bias=qkv_bias)
        self.kv = nn.Linear(dim_kv, dim_q * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim_q, dim_q)
        self.stride = stride
        if stride > 1:
            self.pool = nn.AvgPool2d(stride, stride=stride)
            self.sr = nn.Conv2d(dim_kv, dim_kv, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim_kv)
            self.act = nn.GELU()

    def forward(self, x, y):
        B, N, C = x.shape
        B, L, C2 = y.shape
        H = W = int(math.sqrt(L))
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads
            ).permute(0, 2, 1, 3)
        if self.stride > 1:
            y_ = y.permute(0, 2, 1).contiguous().reshape(B, C2, H, W)
            y_ = self.sr(self.pool(y_)).reshape(B, C2, -1).permute(0, 2, 1)
            y_ = self.norm(y_)
            y_ = self.act(y_)
            kv = self.kv(y_).reshape(B, -1, 2, self.num_heads, C // self.
                num_heads).permute(2, 0, 3, 1, 4).contiguous()
        else:
            kv = self.kv(y).reshape(B, -1, 2, self.num_heads, C // self.
                num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k, v = kv[0], kv[1]
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_q': 4, 'dim_kv': 4}]
