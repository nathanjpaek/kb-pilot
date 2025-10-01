import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data


class Linear(nn.Linear):

    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(Linear, self).__init__(in_dim, out_dim, bias)
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain(
            w_init_gain))


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
        activation='relu'):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = Linear(d_model, dim_feedforward, w_init_gain=activation)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, enc_align = self.self_attn(src, src, src, attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask)
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src, enc_align


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'nhead': 4}]
