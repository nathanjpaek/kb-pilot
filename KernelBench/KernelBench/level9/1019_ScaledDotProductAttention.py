import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, query, key, value, mask=None):
        _1, _2, query_sequence_length, _3 = query.size()
        batch_size, num_head, key_sequence_length, size_per_head = key.size()
        query = query.view(batch_size, num_head, query_sequence_length,
            size_per_head)
        key = key.view(batch_size, num_head, size_per_head, key_sequence_length
            )
        attention_score = torch.einsum('abcd, abde -> abce', query, key)
        attention_score = attention_score / math.sqrt(size_per_head)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -
                1000000000.0)
        attention_score = F.softmax(attention_score, dim=-1)
        result = attention_score @ value
        return result, attention_score


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
