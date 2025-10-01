import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBlock(nn.Module):

    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.1):
        super(MLPBlock, self).__init__()
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.Gelu = nn.GELU()
        if dropout_rate > 0:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)
        else:
            self.dropout1 = None
            self.dropout2 = None

    def forward(self, x):
        out = self.fc1(x)
        out = self.Gelu(out)
        if self.dropout1:
            out = self.dropout1(out)
        out = self.fc2(out)
        if self.dropout2:
            out = self.dropout2(out)
        return out


class MatrixGeneral(nn.Module):

    def __init__(self, in_dim=(768,), feat_dim=(12, 64)):
        super(MatrixGeneral, self).__init__()
        self.weight = nn.Parameter(torch.randn(*in_dim, *feat_dim))
        self.bias = nn.Parameter(torch.zeros(*feat_dim))

    def forward(self, x, dims):
        feat = torch.tensordot(x, self.weight, dims=dims) + self.bias
        return feat


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, in_dim, heads=8, dropout_rate=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.heads = heads
        self.head_dim = in_dim // heads
        self.scale = self.head_dim ** 0.5
        self.query = MatrixGeneral((in_dim,), (self.heads, self.head_dim))
        self.key = MatrixGeneral((in_dim,), (self.heads, self.head_dim))
        self.value = MatrixGeneral((in_dim,), (self.heads, self.head_dim))
        self.out = MatrixGeneral((self.heads, self.head_dim), (in_dim,))
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        _b, _n, _ = x.shape
        q = self.query(x, dims=([2], [0]))
        k = self.key(x, dims=([2], [0]))
        v = self.value(x, dims=([2], [0]))
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        attention_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_weights, dim=-1)
        out = torch.matmul(attention_weights, v)
        out = out.permute(0, 2, 1, 3)
        out = self.out(out, dims=([2, 3], [0, 1]))
        return out


class EncoderBasicBlock(nn.Module):

    def __init__(self, in_dim, mlp_dim, num_heads, dropout_rate=0.1,
        attention_dropout=0.1):
        super(EncoderBasicBlock, self).__init__()
        self.layer_norm1 = nn.LayerNorm(in_dim)
        self.multi_head_att = MultiHeadSelfAttention(in_dim, heads=
            num_heads, dropout_rate=attention_dropout)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        self.layer_norm2 = nn.LayerNorm(in_dim)
        self.mlp = MLPBlock(in_dim, mlp_dim, in_dim, dropout_rate)

    def forward(self, x):
        residual = x
        out = self.layer_norm1(x)
        out = self.multi_head_att(out)
        if self.dropout:
            out = self.dropout(out)
        out += residual
        residual = out
        out = self.layer_norm2(out)
        out = self.mlp(out)
        out += residual
        return out


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'mlp_dim': 4, 'num_heads': 4}]
