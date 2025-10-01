import torch
import numpy as np
import torch.nn as nn


class SoftAttention(nn.Module):
    """
    https://arxiv.org/abs/1803.10916
    """

    def __init__(self, emb_dim, attn_dim):
        super().__init__()
        self.attn_dim = attn_dim
        self.emb_dim = emb_dim
        self.W = torch.nn.Linear(self.emb_dim, self.attn_dim)
        self.v = nn.Parameter(torch.Tensor(self.attn_dim), requires_grad=True)
        stdv = 1.0 / np.sqrt(self.attn_dim)
        for weight in self.v:
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, values):
        attention_weights = self._get_weights(values)
        values = values.transpose(1, 2)
        weighted = torch.mul(values, attention_weights.unsqueeze(1).
            expand_as(values))
        representations = weighted.sum(2).squeeze()
        return representations

    def _get_weights(self, values):
        values.size(0)
        weights = self.W(values)
        weights = torch.tanh(weights)
        e = weights @ self.v
        attention_weights = torch.softmax(e.squeeze(1), dim=-1)
        return attention_weights


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'emb_dim': 4, 'attn_dim': 4}]
