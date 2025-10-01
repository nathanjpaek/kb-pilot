import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectMultiheadAttention(nn.Module):

    def __init__(self, d_in, d_out, heads, dropout=0.1):
        super(DirectMultiheadAttention, self).__init__()
        self.heads = heads
        self.proj_pair = nn.Linear(d_in, heads)
        self.drop = nn.Dropout(dropout, inplace=True)
        self.proj_msa = nn.Linear(d_out, d_out)
        self.proj_out = nn.Linear(d_out, d_out)

    def forward(self, src, tgt):
        B, N, L = tgt.shape[:3]
        attn_map = F.softmax(self.proj_pair(src), dim=1).permute(0, 3, 1, 2)
        attn_map = self.drop(attn_map).unsqueeze(1)
        value = self.proj_msa(tgt).permute(0, 3, 1, 2).contiguous().view(B,
            -1, self.heads, N, L)
        tgt = torch.matmul(value, attn_map).view(B, -1, N, L).permute(0, 2,
            3, 1)
        tgt = self.proj_out(tgt)
        return tgt


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_in': 4, 'd_out': 4, 'heads': 4}]
