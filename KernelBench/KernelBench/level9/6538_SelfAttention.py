from _paritybench_helpers import _mock_config
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class SelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.qkv = nn.Linear(config.n_embd, config.n_embd * 3)
        self.split_size = config.n_embd
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer('mask', torch.tril(torch.ones(config.
            block_size, config.block_size)).view(1, 1, config.block_size,
            config.block_size))
        self.n_head = config.n_head

    def forward(self, x, attn_mask=None):
        B, T, C = x.size()
        q, k, v = self.qkv(x).split(self.split_size, 2)
        att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
        if attn_mask is not None:
            att = att + attn_mask
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        return y, att


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(n_embd=4, n_head=4, block_size=4)}]
