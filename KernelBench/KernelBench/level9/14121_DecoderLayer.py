import torch
from torch import nn


class Ffn(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None,
        act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
        attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, task_embed=None, level=0):
        N, L, D = x.shape
        qkv = self.qkv(x).reshape(N, L, 3, self.num_heads, D // self.num_heads
            ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if task_embed is not None:
            _N, _H, _L, _D = q.shape
            task_embed = task_embed.reshape(1, _H, _L, _D)
            if level == 1:
                q += task_embed
                k += task_embed
            if level == 2:
                q += task_embed
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(N, L, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DecoderLayer(nn.Module):

    def __init__(self, dim, num_heads, ffn_ratio=4.0, qkv_bias=False,
        qk_scale=None, drop=0.0, attn_drop=0.0, act_layer=nn.GELU,
        norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn1 = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        self.attn2 = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm3 = norm_layer(dim)
        ffn_hidden_dim = int(dim * ffn_ratio)
        self.ffn = Ffn(in_features=dim, hidden_features=ffn_hidden_dim,
            act_layer=act_layer, drop=drop)

    def forward(self, x, task_embed):
        x = x + self.attn1(self.norm1(x), task_embed=task_embed, level=1)
        x = x + self.attn2(self.norm2(x), task_embed=task_embed, level=2)
        x = x + self.ffn(self.norm3(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([1, 4, 4, 1])]


def get_init_inputs():
    return [[], {'dim': 4, 'num_heads': 4}]
