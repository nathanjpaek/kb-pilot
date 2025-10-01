import torch
from torch import nn


class PositionEmbs(nn.Module):

    def __init__(self, num_patches, emb_dim, dropout_rate=0.1):
        super(PositionEmbs, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 2,
            emb_dim))
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        out = x + self.pos_embedding
        if self.dropout:
            out = self.dropout(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 6, 4])]


def get_init_inputs():
    return [[], {'num_patches': 4, 'emb_dim': 4}]
