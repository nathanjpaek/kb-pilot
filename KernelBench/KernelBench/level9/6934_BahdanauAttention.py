import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data
from typing import *
from torch.nn import Parameter
import torch.onnx.operators
import torch.optim
import torch.optim.lr_scheduler


class BaseAttention(nn.Module):
    """Base class for attention layers."""

    def __init__(self, query_dim, value_dim, embed_dim=None):
        super().__init__()
        self.query_dim = query_dim
        self.value_dim = value_dim
        self.embed_dim = embed_dim
        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        pass

    def forward(self, query, value, key_padding_mask=None, state=None):
        raise NotImplementedError


class BahdanauAttention(BaseAttention):
    """ Bahdanau Attention."""

    def __init__(self, query_dim, value_dim, embed_dim, normalize=True):
        super().__init__(query_dim, value_dim, embed_dim)
        self.query_proj = nn.Linear(self.query_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(self.value_dim, embed_dim, bias=False)
        self.v = Parameter(torch.Tensor(embed_dim))
        self.normalize = normalize
        if self.normalize:
            self.b = Parameter(torch.Tensor(embed_dim))
            self.g = Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        self.query_proj.weight.data.uniform_(-0.1, 0.1)
        self.value_proj.weight.data.uniform_(-0.1, 0.1)
        nn.init.uniform_(self.v, -0.1, 0.1)
        if self.normalize:
            nn.init.constant_(self.b, 0.0)
            nn.init.constant_(self.g, math.sqrt(1.0 / self.embed_dim))

    def forward(self, query, value, key_padding_mask=None, state=None):
        projected_query = self.query_proj(query).unsqueeze(0)
        key = self.value_proj(value)
        if self.normalize:
            normed_v = self.g * self.v / torch.norm(self.v)
            attn_scores = (normed_v * torch.tanh(projected_query + key +
                self.b)).sum(dim=2)
        else:
            attn_scores = self.v * torch.tanh(projected_query + key).sum(dim=2)
        if key_padding_mask is not None:
            attn_scores = attn_scores.float().masked_fill_(key_padding_mask,
                float('-inf')).type_as(attn_scores)
        attn_scores = F.softmax(attn_scores, dim=0)
        context = (attn_scores.unsqueeze(2) * value).sum(dim=0)
        next_state = attn_scores
        return context, attn_scores, next_state


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'query_dim': 4, 'value_dim': 4, 'embed_dim': 4}]
