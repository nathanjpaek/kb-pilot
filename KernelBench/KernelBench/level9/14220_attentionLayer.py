import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from itertools import product as product


class attentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.1):
        super(attentionLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, src, tar):
        src = src.transpose(0, 1)
        tar = tar.transpose(0, 1)
        src2 = self.self_attn(tar, src, src, attn_mask=None,
            key_padding_mask=None)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        src = src.transpose(0, 1)
        return src


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'nhead': 4}]
