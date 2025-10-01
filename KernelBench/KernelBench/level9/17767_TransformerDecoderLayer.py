import math
import torch
import warnings
from torch import Tensor
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import torch.nn.functional as F
from typing import Optional
from typing import Tuple
from typing import List


def _in_projection_packed(q: 'Tensor', k: 'Tensor', v: 'Tensor', w:
    'Tensor', b: 'Optional[Tensor]'=None) ->List[Tensor]:
    E = q.size(-1)
    if k is v:
        if q is k:
            return F.linear(q, w, b).chunk(3, dim=-1)
        else:
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            return (F.linear(q, w_q, b_q),) + F.linear(k, w_kv, b_kv).chunk(
                2, dim=-1)
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v,
            w_v, b_v)


def scale_dot_attention(q: 'Tensor', k: 'Tensor', v: 'Tensor', dropout_p:
    'float'=0.0, attn_mask: 'Optional[Tensor]'=None) ->Tuple[Tensor, Tensor]:
    _, _, E = q.shape
    q = q / math.sqrt(E)
    attn = torch.bmm(q, k.transpose(-2, -1))
    if attn_mask is not None:
        attn = attn + attn_mask
    attn = F.softmax(attn, dim=-1)
    if dropout_p:
        attn = F.dropout(attn, p=dropout_p)
    out = torch.bmm(attn, v)
    return out, attn


def multi_head_attention_forward(query: 'Tensor', key: 'Tensor', value:
    'Tensor', num_heads: 'int', in_proj_weight: 'Tensor', in_proj_bias:
    'Optional[Tensor]', dropout_p: 'float', out_proj_weight: 'Tensor',
    out_proj_bias: 'Optional[Tensor]', training: 'bool'=True,
    key_padding_mask: 'Optional[Tensor]'=None, need_weights: 'bool'=True,
    attn_mask: 'Optional[Tensor]'=None, use_separate_proj_weight=None,
    q_proj_weight: 'Optional[Tensor]'=None, k_proj_weight:
    'Optional[Tensor]'=None, v_proj_weight: 'Optional[Tensor]'=None) ->Tuple[
    Tensor, Optional[Tensor]]:
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    head_dim = embed_dim // num_heads
    q, k, v = _in_projection_packed(query, key, value, in_proj_weight,
        in_proj_bias)
    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            warnings.warn(
                'Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.'
                )
            attn_mask = attn_mask
        else:
            assert attn_mask.is_floating_point(
                ) or attn_mask.dtype == torch.bool, f'Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}'
        if attn_mask.dim() == 2:
            correct_2d_size = tgt_len, src_len
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(
                    f'The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.'
                    )
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = bsz * num_heads, tgt_len, src_len
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(
                    f'The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.'
                    )
        else:
            raise RuntimeError(
                f"attn_mask's dimension {attn_mask.dim()} is not supported")
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn(
            'Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.'
            )
        key_padding_mask = key_padding_mask
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len
            ), f'expecting key_padding_mask shape of {bsz, src_len}, but got {key_padding_mask.shape}'
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).expand(
            -1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float('-inf'))
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
        new_attn_mask.masked_fill_(attn_mask, float('-inf'))
        attn_mask = new_attn_mask
    if not training:
        dropout_p = 0.0
    attn_output, attn_output_weights = scale_dot_attention(q, k, v,
        attn_mask, dropout_p)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len,
        bsz, embed_dim)
    attn_output = nn.functional.linear(attn_output, out_proj_weight,
        out_proj_bias)
    if need_weights:
        attn_output_weights = attn_output_weights.view(bsz, num_heads,
            tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None


class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, kdim=
        None, vdim=None, batch_first=False) ->None:
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = (self.kdim == self.embed_dim and self.
            vdim == self.embed_dim)
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim)))
            self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim)))
            self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim)))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty((3 * embed_dim,
                embed_dim)))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)
        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)

    def forward(self, query: 'Tensor', key: 'Tensor', value: 'Tensor',
        key_padding_mask: 'Optional[Tensor]'=None, need_weights: 'bool'=
        True, attn_mask: 'Optional[Tensor]'=None) ->Tuple[Tensor, Optional[
        Tensor]]:
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)
                ]
        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.num_heads, self.in_proj_weight,
                self.in_proj_bias, self.dropout, self.out_proj.weight, self
                .out_proj.bias, training=self.training, key_padding_mask=
                key_padding_mask, need_weights=need_weights, attn_mask=
                attn_mask, use_separate_proj_weight=True, q_proj_weight=
                self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.num_heads, self.in_proj_weight,
                self.in_proj_bias, self.dropout, self.out_proj.weight, self
                .out_proj.bias, training=self.training, key_padding_mask=
                key_padding_mask, need_weights=need_weights, attn_mask=
                attn_mask)
        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
        activation=F.relu, layer_norm_eps=1e-05, batch_first=False) ->None:
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout,
            batch_first=batch_first)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=
            dropout, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, tgt: 'Tensor', memory: 'Tensor', tgt_mask:
        'Optional[Tensor]'=None, memory_mask: 'Optional[Tensor]'=None,
        tgt_key_padding_mask: 'Optional[Tensor]'=None,
        memory_key_padding_mask: 'Optional[Tensor]'=None) ->Tensor:
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=
            memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'nhead': 4}]
