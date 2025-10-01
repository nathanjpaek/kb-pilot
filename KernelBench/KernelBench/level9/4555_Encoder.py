import torch
from torch import nn


class Encoder(nn.Module):

    def __init__(self, embedding_dim, nhead, dropout, k=4):
        super(Encoder, self).__init__()
        self.transformer = nn.TransformerEncoderLayer(embedding_dim, nhead,
            dim_feedforward=k * embedding_dim, dropout=dropout, activation=
            'gelu')

    def forward(self, x):
        x = x.transpose(0, 1)
        h = self.transformer(x)
        out = h.mean(dim=0)
        return out


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'embedding_dim': 4, 'nhead': 4, 'dropout': 0.5}]
