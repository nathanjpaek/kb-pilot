import torch
import torch.nn.functional as F
import torch.nn as nn


class AttnLayer(nn.Module):
    """Attention layer.
    w is context vector.
    Formula:
        $$
        v_i=tanh(Wh_i+b)\\
        lpha_i = v_i^Tw\\
        lpha_i = softmax(lpha_i)\\
        Vec = \\sum_0^L lpha_ih_i
        $$
    """

    def __init__(self, hidden_dim, attn_dim):
        super().__init__()
        self.weight = nn.Linear(hidden_dim, attn_dim)
        self.context = nn.Parameter(torch.randn(attn_dim))

    def forward(self, x):
        """
        x: shape=(batch_size, max_len, hidden_dim)
        """
        query = self.weight(x).tanh()
        scores = torch.einsum('bld,d->bl', query, self.context)
        scores = F.softmax(scores, dim=-1)
        attn_vec = torch.einsum('bl,blh->bh', scores, x)
        return attn_vec


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_dim': 4, 'attn_dim': 4}]
