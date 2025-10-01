import math
import torch
from torch import nn
from torch.nn import Linear
import torch as t
from torch.autograd import Variable


class MultiheadAttention(nn.Module):
    """Multihead attention mechanism (dot attention)."""

    def __init__(self, num_hidden_k):
        """:param num_hidden_k: dimension of hidden."""
        super(MultiheadAttention, self).__init__()
        self.num_hidden_k = num_hidden_k
        self.attn_dropout = nn.Dropout(p=0.1)

    def forward(self, key, value, query, mask=None, query_mask=None,
        gaussian_factor=None):
        """forward."""
        attn = t.bmm(query, key.transpose(1, 2))
        attn = attn / math.sqrt(self.num_hidden_k)
        if gaussian_factor is not None:
            attn = attn - gaussian_factor
        if mask is not None:
            attn = attn.masked_fill(mask, -2 ** 32 + 1)
            attn = t.softmax(attn, dim=-1)
        else:
            attn = t.softmax(attn, dim=-1)
        if query_mask is not None:
            attn = attn * query_mask
        result = t.bmm(attn, value)
        return result, attn


class Attention(nn.Module):
    """Attention Network."""

    def __init__(self, num_hidden, h=4, local_gaussian=False):
        """init."""
        super(Attention, self).__init__()
        self.num_hidden = num_hidden
        self.num_hidden_per_attn = num_hidden // h
        self.h = h
        self.key = Linear(num_hidden, num_hidden, bias=False)
        self.value = Linear(num_hidden, num_hidden, bias=False)
        self.query = Linear(num_hidden, num_hidden, bias=False)
        self.multihead = MultiheadAttention(self.num_hidden_per_attn)
        self.local_gaussian = local_gaussian
        if local_gaussian:
            self.local_gaussian_factor = Variable(t.tensor(30.0),
                requires_grad=True).float()
        else:
            self.local_gaussian_factor = None
        self.residual_dropout = nn.Dropout(p=0.1)
        self.final_linear = Linear(num_hidden * 2, num_hidden)
        self.layer_norm_1 = nn.LayerNorm(num_hidden)

    def forward(self, memory, decoder_input, mask=None, query_mask=None):
        """forward."""
        batch_size = memory.size(0)
        seq_k = memory.size(1)
        seq_q = decoder_input.size(1)
        if query_mask is not None:
            query_mask = query_mask.unsqueeze(-1).repeat(1, 1, seq_k)
            query_mask = query_mask.repeat(self.h, 1, 1)
        if mask is not None:
            mask = mask.repeat(self.h, 1, 1)
        key = self.key(memory).view(batch_size, seq_k, self.h, self.
            num_hidden_per_attn)
        value = self.value(memory).view(batch_size, seq_k, self.h, self.
            num_hidden_per_attn)
        query = self.query(decoder_input).view(batch_size, seq_q, self.h,
            self.num_hidden_per_attn)
        key = key.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.
            num_hidden_per_attn)
        value = value.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self
            .num_hidden_per_attn)
        query = query.permute(2, 0, 1, 3).contiguous().view(-1, seq_q, self
            .num_hidden_per_attn)
        if self.local_gaussian:
            row = t.arange(1, seq_k + 1).unsqueeze(0).unsqueeze(-1).repeat(
                batch_size * self.h, 1, seq_k)
            col = t.arange(1, seq_k + 1).unsqueeze(0).unsqueeze(0).repeat(
                batch_size * self.h, seq_k, 1)
            local_gaussian = t.pow(row - col, 2).float()
            self.local_gaussian_factor = self.local_gaussian_factor
            local_gaussian = local_gaussian / self.local_gaussian_factor
        else:
            local_gaussian = None
        result, attns = self.multihead(key, value, query, mask=mask,
            query_mask=query_mask, gaussian_factor=local_gaussian)
        attns = attns.view(self.h, batch_size, seq_q, seq_k)
        attns = attns.permute(1, 0, 2, 3)
        result = result.view(self.h, batch_size, seq_q, self.
            num_hidden_per_attn)
        result = result.permute(1, 2, 0, 3).contiguous().view(batch_size,
            seq_q, -1)
        result = t.cat([decoder_input, result], dim=-1)
        result = self.final_linear(result)
        result = result + decoder_input
        result = self.layer_norm_1(result)
        return result, attns


class Linear(nn.Module):
    """Linear Module."""

    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        """init."""
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.linear_layer.weight, gain=nn.init.
            calculate_gain(w_init))

    def forward(self, x):
        """forward."""
        return self.linear_layer(x)


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'num_hidden': 4}]
