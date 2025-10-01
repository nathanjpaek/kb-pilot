import math
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):

    def __init__(self, heads, d_model):
        super(MultiHeadAttention, self).__init__()
        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.heads = heads
        self.dropout = nn.Dropout(0.1)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.concat = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask):
        """
        query, key, value of shape: (batch_size, max_len, 512)
        mask of shape: (batch_size, 1, 1, max_words)
        """
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)
        query = query.view(query.shape[0], -1, self.heads, self.d_k).permute(
            0, 2, 1, 3)
        key = key.view(key.shape[0], -1, self.heads, self.d_k).permute(0, 2,
            1, 3)
        value = value.view(value.shape[0], -1, self.heads, self.d_k).permute(
            0, 2, 1, 3)
        scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / math.sqrt(query
            .size(-1))
        scores = scores.masked_fill(mask == 0, -1000000000.0)
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        context = torch.matmul(weights, value)
        context = context.permute(0, 2, 1, 3).contiguous().view(context.
            shape[0], -1, self.heads * self.d_k)
        interacted = self.concat(context)
        return interacted


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 16, 16])]


def get_init_inputs():
    return [[], {'heads': 4, 'd_model': 4}]
