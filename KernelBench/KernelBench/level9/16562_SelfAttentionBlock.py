import torch
import torch.distributed
import torch
import torch.nn as nn
import torch.nn.functional
import torch.utils.data
import torch.optim
import torch.optim.lr_scheduler


class Mlp(nn.Module):
    """ Multilayer perceptron."""

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
        """
            Args:
                x (torch.Tensor): (B, L, C), input tensor
            Returns:
                torch.Tensor: (B, L, C), output tensor
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SelfAttention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
        attn_drop=0.0, proj_drop=0.0, attn_pos_encoding_only=False):
        super(SelfAttention, self).__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by num_heads {num_heads}.'
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        if attn_pos_encoding_only:
            self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        else:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.k = nn.Linear(dim, dim, bias=qkv_bias)
            self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_pos_encoding_only = attn_pos_encoding_only

    def forward(self, x, q_ape, k_ape, attn_pos):
        """
            Args:
                x (torch.Tensor): (B, L, C)
                q_ape (torch.Tensor | None): (1 or B, L, C), absolute positional encoding for q
                k_ape (torch.Tensor | None): (1 or B, L, C), absolute positional encoding for k
                attn_pos (torch.Tensor | None): (1 or B, num_heads, L, L), untied positional encoding
            Returns:
                torch.Tensor: (B, L, C)
        """
        B, N, C = x.shape
        if self.attn_pos_encoding_only:
            assert q_ape is None and k_ape is None
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.
                num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            q = x + q_ape if q_ape is not None else x
            q = self.q(q).reshape(B, N, self.num_heads, C // self.num_heads
                ).permute(0, 2, 1, 3)
            k = x + k_ape if k_ape is not None else x
            k = self.k(k).reshape(B, -1, self.num_heads, C // self.num_heads
                ).permute(0, 2, 1, 3)
            v = self.v(x).reshape(B, -1, self.num_heads, C // self.num_heads
                ).permute(0, 2, 1, 3)
        attn = q @ k.transpose(-2, -1)
        attn = attn * self.scale
        if attn_pos is not None:
            attn = attn + attn_pos
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SelfAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False,
        qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=nn.Identity(),
        act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_pos_encoding_only=
        False):
        super(SelfAttentionBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SelfAttention(dim, num_heads, qkv_bias, qk_scale,
            attn_drop, drop, attn_pos_encoding_only)
        self.drop_path = drop_path
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
            act_layer=act_layer, drop=drop)

    def forward(self, x, q_ape, k_ape, attn_pos):
        """
            Args:
                x (torch.Tensor): (B, L, C)
                q_ape (torch.Tensor | None): (1 or B, L, C), absolute positional encoding for q
                k_ape (torch.Tensor | None): (1 or B, L, C), absolute positional encoding for k
                attn_pos (torch.Tensor | None): (1 or B, num_heads, L, L), untied positional encoding
            Returns:
                torch.Tensor: (B, L, C)
        """
        x = x + self.drop_path(self.attn(self.norm1(x), q_ape, k_ape, attn_pos)
            )
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4]), torch.rand([4, 4]),
        torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'dim': 4, 'num_heads': 4}]
