import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, attn_mask: 'torch.Tensor'=None, valid_input_mask:
        'torch.Tensor'=None, mask_value=-1000000.0):
        """mask should be a 3D tensor of shape (B, T, T)"""
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2)
        att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
        if attn_mask is not None:
            att = att.masked_fill(attn_mask.unsqueeze(1) == 0, mask_value)
        if valid_input_mask is not None:
            att = att.masked_fill(valid_input_mask.unsqueeze(1).unsqueeze(2
                ) == 0, mask_value)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'n_embd': 4, 'n_head': 4, 'attn_pdrop': 0.5, 'resid_pdrop':
        0.5}]
