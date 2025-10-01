import torch
import torch.nn as nn
from torch.nn import functional as F
import torch._utils


class MHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads=1, dropout=0.0,
        bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k):
        q = self.q_linear(q)
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1),
            self.k_linear.bias)
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim //
            self.num_heads)
        kh = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.
            num_heads, k.shape[-2], k.shape[-1])
        weights = torch.einsum('bqnc,bnchw->bqnhw', qh * self.
            normalize_fact, kh)
        weights = F.softmax(weights.flatten(2), dim=-1).view_as(weights)
        weights = self.dropout(weights)
        return weights


def get_inputs():
    return [torch.rand([4, 4, 1, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'query_dim': 4, 'hidden_dim': 4}]
