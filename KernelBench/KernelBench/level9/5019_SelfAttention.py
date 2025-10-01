import torch
import torch.nn as nn
import torch.distributed


class SelfAttention(nn.Module):

    def __init__(self, model_dim, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.Va = nn.Linear(model_dim, 1, bias=False)
        self.Wa = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b, t, n = x.size()
        proj = torch.tanh(self.Wa(x.view(b * t, n).contiguous()))
        scores = self.Va(proj)
        scores = scores.view(b, t).contiguous()
        if mask is not None:
            scores = scores.masked_fill(mask, -1e+18)
        attn = torch.softmax(scores, -1)
        drop_attn = self.dropout(attn)
        context = torch.sum(drop_attn.unsqueeze(-1) * x, -2)
        return context, attn


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'model_dim': 4}]
