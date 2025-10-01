import math
import torch
import torch.nn as nn


class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, hidden_dim, n_heads, dropout=0.1):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)
        self.fc_o = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-06)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q = self.layer_norm(q)
        q = self.fc_q(q)
        k = self.fc_k(k)
        v = self.fc_v(v)
        q = q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 
            2, 1, 3)
        k = k.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 
            2, 1, 3)
        v = v.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 
            2, 1, 3)
        att = torch.matmul(q / self.scale, k.permute(0, 1, 3, 2))
        if mask is not None:
            att = att.masked_fill(mask == 0, -10000000000.0)
        att = torch.softmax(att, dim=-1)
        out = torch.matmul(self.dropout(att), v)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, self.hidden_dim)
        out = self.dropout(self.fc_o(out))
        return out, att


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4,
        4, 4])]


def get_init_inputs():
    return [[], {'hidden_dim': 4, 'n_heads': 4}]
