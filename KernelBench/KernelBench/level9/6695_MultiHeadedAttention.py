import math
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim


class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from
    "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.

    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.

    .. mermaid::

       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O

    Also includes several additional tricks.

    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, head_count, model_dim, dropout=0.1):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim
        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count
        self.linear_keys = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim, head_count * self.
            dim_per_head)
        self.linear_query = nn.Linear(model_dim, head_count * self.dim_per_head
            )
        self.sm = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)

    def _split_heads(self, x):
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.head_count, self.dim_per_head
            ).transpose(1, 2)

    def _combine_heads(self, x):
        """:param x: [batch_size * head_count, seq_len, dim_per_head]"""
        seq_len = x.size(2)
        return x.transpose(1, 2).contiguous().view(-1, seq_len, self.
            head_count * self.dim_per_head)

    def compute_cache(self, key, value):
        key_up = self._split_heads(self.linear_keys(key))
        value_up = self._split_heads(self.linear_values(value))
        return [key_up, value_up]

    def forward(self, key, value, query, mask=None, cache=None):
        """
        Compute the context vector and the attention vectors.

        Args:
           key (`FloatTensor`): set of `key_len`
                key vectors `[batch, key_len, dim]`
           value (`FloatTensor`): set of `key_len`
                value vectors `[batch, key_len, dim]`
           query (`FloatTensor`): set of `query_len`
                 query vectors  `[batch, query_len, dim]`
           mask: binary mask indicating which keys have
                 non-zero attention `[batch, query_len, key_len]`
        Returns:
           (`FloatTensor`, `FloatTensor`) :

           * output context vectors `[batch, query_len, dim]`
           * one of the attention vectors `[batch, query_len, key_len]`
        """
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)
        if cache is not None:
            key_up, value_up = cache
        else:
            key_up = self._split_heads(self.linear_keys(key))
            value_up = self._split_heads(self.linear_values(value))
        query_up = self._split_heads(self.linear_query(query))
        query_up = query_up / math.sqrt(dim_per_head)
        scores = torch.matmul(query_up, key_up.transpose(2, 3))
        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(Variable(mask), -1e+18)
        attn = self.sm(scores)
        drop_attn = self.dropout(attn)
        context = self._combine_heads(torch.matmul(drop_attn, value_up))
        output = self.final_linear(context)
        top_attn = attn.view(batch_size, head_count, query_len, key_len)[:,
            0, :, :].contiguous()
        return output, top_attn


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'head_count': 4, 'model_dim': 4}]
