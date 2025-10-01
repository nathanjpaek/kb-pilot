import math
import torch
import torch.nn as nn
from typing import Optional
from typing import List


class FeedForward(nn.Module):
    """
    ## FFN module
    """

    def __init__(self, d_model: 'int', d_ff: 'int', dropout: 'float'=0.1,
        activation=nn.ReLU(), is_gated: 'bool'=False, bias: 'bool'=True,
        bias_gate: 'bool'=True):
        """
        * d_model is the number of features
        * d_ff is the number of features in the hidden layer of the FFN
        * dropout is dropout probability for the hidden layer
        * is_gated specifies whether the hidden layer is gated
        * bias1 specified whether the first fully connected layer should have a learnable bias
        * bias2 specified whether the second fully connected layer should have a learnable bias
        * bias_gate specified whether the fully connected layer for the gate should have a learnable bias
        """
        super(FeedForward, self).__init__()
        self.layer1 = nn.Linear(d_model, d_ff, bias=bias)
        self.layer2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.is_gated = is_gated
        if is_gated:
            self.linear_v = nn.Linear(d_model, d_ff, bias=bias_gate)

    def forward(self, x: 'torch.Tensor'):
        g = self.activation(self.layer1(x))
        if self.is_gated:
            x = g * self.linear_v(x)
        else:
            x = g
        x = self.dropout(x)
        return self.layer2(x)


class PrepareForMultiHeadAttention(nn.Module):
    """
    ## Prepare for multi-head attention
    这个linear transform作用是把query，key，value映射到同一个低维空间内
    This module does a linear transformation and splits the vector into given
    number of heads for multi-head attention.
    This is used to transform **key**, **query**, and **value** vectors.
    """

    def __init__(self, d_model: 'int', heads: 'int', d_k: 'int', bias: 'bool'):
        super(PrepareForMultiHeadAttention, self).__init__()
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        self.heads = heads
        self.d_k = d_k

    def forward(self, x: 'torch.Tensor'):
        head_shape = x.shape[:-1]
        x = self.linear(x)
        x = x.view(*head_shape, self.heads, self.d_k)
        return x


class MultiHeadAttention(nn.Module):
    """
    This computes scaled multi-headed attention for given query, key and value vectors.

    compute similatiry between query and key, use this as attention efficient multiply value

    It uses dot-product of query and key as the indicator of how matching they are.
    Before taking the $softmax$ the dot-products are scaled by $\\frac{1}{\\sqrt{d_k}}$.
    This is done to avoid large dot-product values causing softmax to
    give very small gradients when $d_k$ is large.

    Softmax is calculated along the axis of of the sequence (or time).
    """

    def __init__(self, heads: 'int', d_model: 'int', dropout_prob: 'float'=
        0.1, bias: 'bool'=True):
        """
        * heads is the number of heads.
        * d_model is the number of features in the query, key and value vectors.
        """
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_model // heads
        self.heads = heads
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k,
            bias=bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k,
            bias=bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k,
            bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_prob)
        self.scale = 1 / math.sqrt(self.d_k)
        self.attn = None

    def get_scores(self, query: 'torch.Tensor', key: 'torch.Tensor'):
        """
        ### Calculate scores between queries and keys，使用的是点积的方法
        还可以有cosine，MLP等计算相似度的方法
        """
        return torch.einsum('ibhd,jbhd->ijbh', query, key)

    def prepare_mask(self, mask: 'torch.Tensor', query_shape: 'List[int]',
        key_shape: 'List[int]'):
        """
        mask has shape [seq_len_q, seq_len_k, batch_size], where first dimension is the query dimension.
        If the query dimension is equal to $1$ it will be broadcasted.
        """
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]
        mask = mask.unsqueeze(-1)
        return mask

    def forward(self, *, query: torch.Tensor, key: torch.Tensor, value:
        torch.Tensor, mask: Optional[torch.Tensor]=None):
        seq_len, batch_size, _ = query.shape
        if mask is not None:
            mask = self.prepare_mask(mask, query.shape, key.shape)
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)
        scores = self.get_scores(query, key)
        scores = scores * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = self.softmax(scores)
        attn = self.dropout(attn)
        x = torch.einsum('ijbh,jbhd->ibhd', attn, value)
        self.attn = attn.detach()
        x = x.reshape(seq_len, batch_size, -1)
        return self.output(x)


class EncoderLayer(nn.Module):

    def __init__(self, d_model: 'int', d_ff: 'int', heads: 'int', bias:
        'bool'=True, is_gated: 'bool'=False, bias_gate: 'bool'=True,
        activation=nn.ELU(), dropout_prob: 'float'=0.1):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(heads, d_model, dropout_prob, bias)
        self.feed_forward = FeedForward(d_model, d_ff, dropout_prob,
            activation, is_gated, bias, bias_gate)
        self.dropout = nn.Dropout(dropout_prob)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        z = self.norm1(x)
        a = self.attn(query=z, key=z, value=z)
        x = x + self.dropout(a)
        z = self.norm2(x)
        a = self.feed_forward(z)
        x = x + self.dropout(a)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'd_ff': 4, 'heads': 4}]
