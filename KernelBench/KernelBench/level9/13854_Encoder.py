import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch.optim
import torch._utils
import torch.nn


def drop_path(x, drop_prob: 'float'=0.0, training: 'bool'=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.
        device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


def activation(act_type='swish'):
    if act_type == 'swish':
        act = swish()
        return act
    else:
        act = nn.ReLU(inplace=True)
        return act


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class swish(nn.Module):

    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x


class GELU(nn.Module):

    @staticmethod
    def forward(x):
        erf = F.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3)))
        return 0.5 * x * (1 + erf)


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.1, activation=GELU):
        super(FeedForward, self).__init__()
        self.mlp1 = nn.Linear(dim, hidden_dim)
        self.act = activation()
        self.mlp2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.mlp1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.mlp2(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, dim, heads=8, dropout=0.1, attention_dropout=0.1,
        qkv_bias=True):
        super(MultiHeadAttention, self).__init__()
        assert dim % heads == 0
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.to_out = nn.Linear(dim, dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.heads, C // self.heads
            ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attention_dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.to_out(x)
        x = self.dropout(x)
        return x


class Encoder1DBlock(nn.Module):

    def __init__(self, hidden_dim, mlp_dim, heads, dropout,
        attention_dropout, drop_path, qkv_bias, activation, norm_layer=nn.
        LayerNorm):
        super(Encoder1DBlock, self).__init__()
        self.norm1 = norm_layer(hidden_dim)
        self.attention = MultiHeadAttention(hidden_dim, heads, dropout,
            attention_dropout, qkv_bias)
        self.norm2 = norm_layer(hidden_dim)
        self.feedforward = FeedForward(hidden_dim, mlp_dim, dropout,
            activation=activation)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else None

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.attention(x)
        if self.drop_path is not None:
            x = self.drop_path(x)
        x = x + residual
        y = self.norm2(x)
        y = self.feedforward(y)
        if self.drop_path is not None:
            y = self.drop_path(y)
        return x + y


class Encoder(nn.Module):

    def __init__(self, hidden_dim, depth, mlp_dim, heads, dropout=0.1,
        attention_dropout=0.1, drop_path=0.1, qkv_bias=True, activation=GELU):
        super(Encoder, self).__init__()
        encoder_layer = OrderedDict()
        for d in range(depth):
            encoder_layer['encoder_{}'.format(d)] = Encoder1DBlock(hidden_dim,
                mlp_dim, heads, dropout, attention_dropout, drop_path,
                qkv_bias, activation)
        self.encoders = nn.Sequential(encoder_layer)
        self.encoder_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.encoders(x)
        x = self.encoder_norm(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_dim': 4, 'depth': 1, 'mlp_dim': 4, 'heads': 4}]
