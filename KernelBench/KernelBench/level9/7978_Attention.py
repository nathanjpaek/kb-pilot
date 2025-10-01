import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class Linear(nn.Module):
    """
  Linear Module
  """

    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        """
    :param in_dim: dimension of input
    :param out_dim: dimension of output
    :param bias: boolean. if True, bias is included.
    :param w_init: str. weight inits with xavier initialization.
    """
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.linear_layer.weight, gain=nn.init.
            calculate_gain(w_init))

    def forward(self, x):
        return self.linear_layer(x)


class MultiheadAttention(nn.Module):
    """
  Multihead attention mechanism (dot attention)
  """

    def __init__(self, num_hidden_k, dropout_p=0.1):
        """
    :param num_hidden_k: dimension of hidden 
    """
        super(MultiheadAttention, self).__init__()
        self.num_hidden_k = num_hidden_k
        self.attn_dropout = nn.Dropout(p=dropout_p)

    def forward(self, key, value, query, mask=None):
        attn = torch.matmul(query, key.transpose(2, 3))
        attn = attn / math.sqrt(self.num_hidden_k)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1000000000.0)
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        result = torch.matmul(attn, value)
        return result, attn


class Attention(nn.Module):
    """
  Attention Layer used in Tranformer
  """

    def __init__(self, num_hidden, h=4):
        """
    :param num_hidden: dimension of hidden
    :param h: num of heads 
    """
        super(Attention, self).__init__()
        self.num_hidden = num_hidden
        self.num_hidden_per_attn = num_hidden // h
        self.h = h
        self.key = Linear(num_hidden, num_hidden, bias=False)
        self.value = Linear(num_hidden, num_hidden, bias=False)
        self.query = Linear(num_hidden, num_hidden, bias=False)
        self.multihead = MultiheadAttention(self.num_hidden_per_attn)
        self.residual_dropout = nn.Dropout(p=0.1)
        self.final_linear = Linear(num_hidden * 2, num_hidden)
        self.layer_norm = nn.LayerNorm(num_hidden)

    def forward(self, key, value, query, mask=None):
        batch_size = key.size(0)
        seq_k = key.size(1)
        seq_q = query.size(1)
        seq_v = value.size(1)
        residual = value
        key = self.key(key).view(batch_size, seq_k, self.h, self.
            num_hidden_per_attn)
        value = self.value(value).view(batch_size, seq_v, self.h, self.
            num_hidden_per_attn)
        query = self.query(query).view(batch_size, seq_q, self.h, self.
            num_hidden_per_attn)
        query, key, value = query.transpose(1, 2), key.transpose(1, 2
            ), value.transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
        result, attns = self.multihead(key, value, query, mask=mask)
        result = result.transpose(1, 2).contiguous().view(batch_size, seq_k, -1
            )
        result = torch.cat([residual, result], dim=-1)
        result = F.relu(self.final_linear(result))
        result = self.residual_dropout(result)
        result = result + residual
        result = self.layer_norm(result)
        return result, attns


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'num_hidden': 4}]
