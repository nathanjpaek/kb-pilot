import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F
import torch.hub


class Attention(nn.Module):

    def forward(self, query, key, value, mask=None, dropout=None):
        scale = query.size(-1) ** -0.5
        scores = query.matmul(key.transpose(-2, -1)) / scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1000000000.0)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return p_attn.matmul(value), p_attn


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
