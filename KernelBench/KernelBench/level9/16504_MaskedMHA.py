import math
import torch
import torch.nn as nn
import torch.utils.data
from torch.nn import functional as F


class MaskedMHA(nn.Module):
    """
    Multi Head Attention with mask

    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(self, n_embd, n_head, attn_pdrop=0.0, proj_pdrop=0.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)
        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

    def forward(self, x, mask):
        B, C, _T = x.size()
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        att = q * self.scale @ k.transpose(-2, -1)
        att = att.masked_fill(torch.logical_not(mask[:, :, None, :]), float
            ('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        out = att @ (v * mask[:, :, :, None].float())
        out = out.transpose(2, 3).contiguous().view(B, C, -1)
        out = self.proj_drop(self.proj(out)) * mask.float()
        return out, mask


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'n_embd': 4, 'n_head': 4}]
