import math
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""

    def __init__(self, features, eps=1e-06):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class GELU(nn.Module):

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 
            0.044715 * torch.pow(x, 3))))


class FeedForwardNetwork(nn.Module):

    def __init__(self, in_dim, hid_dim) ->None:
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hid_dim)
        self.lin2 = nn.Linear(hid_dim, in_dim)
        self.gleu = GELU()
        self.dropout = nn.Dropout()

    def forward(self, inputs):
        hid = self.gleu(self.lin1(inputs))
        return self.lin2(self.dropout(hid))


class ResConnectionLayer(nn.Module):

    def __init__(self, in_dim, dropout):
        super(ResConnectionLayer, self).__init__()
        self.norm = LayerNorm(in_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = FeedForwardNetwork(in_dim, in_dim)

    def forward(self, x):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(self.ffn(self.norm(x)))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'dropout': 0.5}]
