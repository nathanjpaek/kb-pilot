import torch
import torch.nn.functional as F
import torch.nn as nn


class Linear(nn.Module):

    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.
            calculate_gain(w_init))

    def forward(self, x):
        return self.linear(x)


class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff=2048, dropout=0.1, activation='relu'):
        super(FeedForward, self).__init__()
        self.linear1 = Linear(d_model, d_ff, w_init=activation)
        self.linear2 = Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        output = self.linear2(F.relu(self.linear1(x)))
        output = self.dropout(output)
        output = x + output
        output = self.norm(output)
        return output


class DecoderLayer(nn.Module):
    """Transformer DecoderLayer"""

    def __init__(self, d_model, n_head, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, enc_outputs, dec_attn_mask=None, enc_padding_mask=
        None, dec_padding_mask=None):
        output, self_attention = self.attn(x, x, x, key_padding_mask=
            dec_padding_mask, attn_mask=dec_attn_mask)
        output, context_attention = self.attn(output, enc_outputs,
            enc_outputs, key_padding_mask=enc_padding_mask)
        output = self.ff(output)
        return output, self_attention, context_attention


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'n_head': 4, 'd_ff': 4, 'dropout': 0.5}]
