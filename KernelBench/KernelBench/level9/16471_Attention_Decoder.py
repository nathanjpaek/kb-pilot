import torch
import torch.nn as nn
import torch._utils


class Attention_Decoder(nn.Module):

    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None,
        attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.fc_q = nn.Linear(dim, dim * 1, bias=qkv_bias)
        self.fc_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, x):
        B, N, C = x.shape
        n_class = q.shape[1]
        q = self.fc_q(q).reshape(B, self.num_heads, n_class, C // self.
            num_heads)
        kv = self.fc_kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads
            ).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn1 = q @ k.transpose(-2, -1) * self.scale
        attn2 = attn1.softmax(dim=-1)
        attn3 = self.attn_drop(attn2)
        x = (attn3 @ v).reshape(B, n_class, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        attn = attn1.permute(0, 2, 1, 3)
        return attn, x


def get_inputs():
    return [torch.rand([4, 1, 1, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
