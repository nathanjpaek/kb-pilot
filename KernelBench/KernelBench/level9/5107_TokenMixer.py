import torch
import torch.nn.functional as F
from torch import nn


class FeedForward(nn.Module):

    def __init__(self, num_features, expansion_factor, dropout):
        super().__init__()
        num_hidden = expansion_factor * num_features
        self.fc1 = nn.Linear(num_features, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_features)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout1(F.gelu(self.fc1(x)))
        x = self.dropout2(self.fc2(x))
        return x


class TokenMixer(nn.Module):

    def __init__(self, d_model, seq_len, expansion_factor, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mlp = FeedForward(seq_len, expansion_factor, dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.mlp(x)
        x = x.transpose(1, 2)
        out = x + residual
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'seq_len': 4, 'expansion_factor': 4,
        'dropout': 0.5}]
