import torch
import torch.nn as nn
from typing import Callable


class MLP(nn.Module):
    """Multi Layer Perceptron class"""

    def __init__(self, in_feats: 'int', hidden_feats: 'int'=None, out_feats:
        'int'=None, act_layer: 'Callable[[torch.Tensor], torch.Tensor]'=nn.
        GELU, drop_rate: 'float'=0.0):
        """Initialize MLP

        Args:
            in_feats (int): input number of features
            hidden_feats (int, optional): hidden dimension. Defaults to None.
            out_feats (int, optional): output dimension. Defaults to None.
            act_layer (Callable[[torch.Tensor], torch.Tensor], optional): activation function.
                                                                          Defaults to nn.GELU.
            drop_rate (float, optional): dropout. Defaults to 0.0.
        """
        super().__init__()
        hidden_feats = hidden_feats or in_feats
        out_feats = out_feats or in_feats
        self.fc1 = nn.Linear(in_feats, hidden_feats)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_feats, out_feats)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Forward"""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Attention class"""

    def __init__(self, dim: 'int', num_heads: 'int'=8, qkv_bias: 'bool'=
        False, qk_scale: 'float'=None, attn_drop: 'float'=0.0, proj_drop:
        'float'=0.0):
        """Initialize module

        Args:
            dim (int): input dimension
            num_heads (int, optional): number of heads. Defaults to 8.
            qkv_bias (bool, optional): Apply bias. Defaults to False.
            qk_scale (float, optional): scale factor to query-key. Defaults to None.
            attn_drop (float, optional): dropout for attention. Defaults to 0.0.
            proj_drop (float, optional): dropout. Defaults to 0.0.
        """
        super().__init__()
        self._num_heads = num_heads
        head_dim = dim // num_heads
        self._scale = head_dim ** -0.5
        self._qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self._attn_drop = nn.Dropout(attn_drop)
        self._proj = nn.Linear(dim, dim)
        self._proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Forward"""
        B, N, C = x.shape
        qkv_out = self._qkv(x).view(B, N, 3, self._num_heads, C // self.
            _num_heads)
        qkv = qkv_out.permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1) * self._scale
        attn = attn.softmax(dim=-1)
        attn = self._attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self._proj(x)
        x = self._proj_drop(x)
        return x


class MultiHeadedAttentionBlock(nn.Module):
    """Multi-headed attention block"""

    def __init__(self, dim: 'int', num_heads: 'int', mlp_ratio: 'float'=4.0,
        qkv_bias: 'bool'=False, qk_scale: 'float'=None, drop: 'float'=0.0,
        attn_drop: 'float'=0.0, dropout_ratio=0.0, act_layer:
        'Callable[[torch.Tensor], torch.Tensor]'=nn.GELU, norm_layer:
        'Callable[[torch.Tensor], torch.Tensor]'=nn.LayerNorm):
        """Initialize class

        Args:
            dim (int): dimension
            num_heads (int): number of heads
            mlp_ratio (float, optional): How much it changes the input. Defaults to 4..
            qkv_bias (bool, optional): Apply bias. Defaults to False.
            qk_scale (float, optional): scale factor. Defaults to None.
            drop (float, optional): dropout for MLP. Defaults to 0.0.
            attn_drop (float, optional): dropout on attention layer. Defaults to 0.0.
            dropout_ratio (float, optional): drop-out for positional embedding.
                                             Defaults to 0.0.
            act_layer (Callable[[torch.Tensor], torch.Tensor], optional): activation layer.
                                                                          Defaults to nn.GELU.
            norm_layer (Callable[[torch.Tensor], torch.Tensor], optional): normalization layer.
                                                                           Defaults to nn.LayerNorm.
        """
        super().__init__()
        self._attn_norm = norm_layer(dim)
        self._mlp_norm = norm_layer(dim)
        self._attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        if dropout_ratio > 0.0:
            self._drop_path = nn.Dropout(p=dropout_ratio, inplace=True)
        else:
            self._drop_path = nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self._mlp = MLP(in_feats=dim, hidden_feats=mlp_hidden_dim,
            act_layer=act_layer, drop_rate=drop)

    def forward(self, x) ->torch.Tensor:
        """Forward"""
        x = x + self._drop_path(self._attn(self._attn_norm(x)))
        x = x + self._drop_path(self._mlp(self._mlp_norm(x)))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4, 'num_heads': 4}]
