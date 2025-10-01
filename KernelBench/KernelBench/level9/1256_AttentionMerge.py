import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMerge(nn.Module):

    def __init__(self, input_size, attention_size, dropout_prob=0.1):
        super(AttentionMerge, self).__init__()
        self.attention_size = attention_size
        self.hidden_layer = nn.Linear(input_size, self.attention_size)
        self.query_ = nn.Parameter(torch.Tensor(self.attention_size, 1))
        self.dropout = nn.Dropout(dropout_prob)
        self.query_.data.normal_(mean=0.0, std=0.02)

    def forward(self, values, mask=None):
        """
        H (B, L, hidden_size) => h (B, hidden_size)
        (B, L1, L2, hidden_size) => (B, L2, hidden)
        """
        if mask is None:
            mask = torch.zeros_like(values)
        else:
            mask = (1 - mask.unsqueeze(-1).type(torch.float)) * -1000.0
        keys = self.hidden_layer(values)
        keys = torch.tanh(keys)
        query_var = torch.var(self.query_)
        attention_probs = keys @ self.query_ / math.sqrt(self.
            attention_size * query_var)
        attention_probs = F.softmax(attention_probs * mask, dim=1)
        attention_probs = self.dropout(attention_probs)
        context = torch.sum(attention_probs + values, dim=1)
        return context


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'attention_size': 4}]
