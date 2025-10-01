import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1000000000.0)
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output


class Norm(nn.Module):

    def __init__(self, d_model, eps=1e-06):
        super().__init__()
        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        x_mean = x.mean(dim=-1, keepdim=True)
        x_variance = x.std(dim=-1, keepdim=True)
        normalized_x = (x - x_mean) / (x_variance + self.eps)
        y = self.alpha * normalized_x + self.bias
        return y


class MultiHeadAttention(nn.Module):

    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output


class FeedForward(nn.Module):

    def __init__(self, d_model: 'int', d_ff: 'int'=2048, dropout_pct:
        'float'=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_pct)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.linear2(x)
        return x


class AddNorm(nn.Module):

    def __init__(self, d_model, dropout_pct=0.1):
        super().__init__()
        self.norm = Norm(d_model)
        self.dropout = nn.Dropout(dropout_pct)

    def forward(self, x, attn_output):
        x = x + self.dropout(attn_output)
        x = self.norm(x)
        return x


class DecoderLayer(nn.Module):

    def __init__(self, d_model, heads, dropout_pct=0.1):
        super().__init__()
        self.attn_decoder = MultiHeadAttention(heads, d_model)
        self.attn_encoder = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.add_norm_1 = AddNorm(d_model, dropout_pct)
        self.add_norm_2 = AddNorm(d_model, dropout_pct)
        self.add_norm_3 = AddNorm(d_model, dropout_pct)

    def forward(self, x, encoder_output, src_mask, trg_mask):
        """
        x: this x comes from the target language
        """
        q = k = v = x
        x = self.add_norm_1(x, self.attn_decoder(q, k, v, trg_mask))
        k_enc = v_enc = encoder_output
        x = self.add_norm_2(x, self.attn_encoder(x, k_enc, v_enc, src_mask))
        x = self.add_norm_3(x, self.ff(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4,
        4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'heads': 4}]
