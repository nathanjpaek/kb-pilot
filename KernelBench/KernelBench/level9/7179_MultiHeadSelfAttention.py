from _paritybench_helpers import _mock_config
import math
import torch
import torch.distributed
import torch.nn.functional as F
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, config):
        super(MultiHeadSelfAttention, self).__init__()
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        assert config.hidden_size % config.n_heads == 0, 'Hidden size should be multiple of n_heads'
        self.n_heads = config.n_heads
        self.head_size = config.hidden_size // self.n_heads

    def forward(self, x):
        batch_size, seq_length, hidden_size = x.size()
        q = self.query(x).view(batch_size, seq_length, self.n_heads, self.
            head_size).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_length, self.head_size, self.
            n_heads).transpose(1, 3)
        v = self.value(x).view(batch_size, seq_length, self.n_heads, self.
            head_size).transpose(1, 2)
        attention_mask = torch.full((seq_length, seq_length), -float('inf'),
            device=x.device, dtype=x.dtype)
        attention_mask = torch.triu(attention_mask, diagonal=1)
        attention_score = torch.matmul(q, k) / math.sqrt(self.head_size
            ) + attention_mask
        attention_score = F.softmax(attention_score, dim=-1)
        attention_score = self.attn_drop(attention_score)
        score = torch.matmul(attention_score, v)
        score = score.transpose(1, 2).contiguous().view(batch_size,
            seq_length, hidden_size)
        score = self.proj(score)
        score = self.resid_drop(score)
        return score


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(hidden_size=4, attn_pdrop=0.5,
        resid_pdrop=0.5, n_heads=4)}]
