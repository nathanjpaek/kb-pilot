import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceWeight(nn.Module):

    def __init__(self, d_model, heads, dropout=0.1):
        super(SequenceWeight, self).__init__()
        self.heads = heads
        self.d_model = d_model
        self.d_k = d_model // heads
        self.scale = 1.0 / math.sqrt(self.d_k)
        self.to_query = nn.Linear(d_model, d_model)
        self.to_key = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, msa):
        B, N, L = msa.shape[:3]
        msa = msa.permute(0, 2, 1, 3)
        tar_seq = msa[:, :, 0].unsqueeze(2)
        q = self.to_query(tar_seq).view(B, L, 1, self.heads, self.d_k).permute(
            0, 1, 3, 2, 4).contiguous()
        k = self.to_key(msa).view(B, L, N, self.heads, self.d_k).permute(0,
            1, 3, 4, 2).contiguous()
        q = q * self.scale
        attn = torch.matmul(q, k)
        attn = F.softmax(attn, dim=-1)
        return self.dropout(attn)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'heads': 4}]
