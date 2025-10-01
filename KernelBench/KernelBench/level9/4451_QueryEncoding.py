import torch
import torch.nn as nn


class QueryEncoding(nn.Module):

    def __init__(self, d_model):
        super(QueryEncoding, self).__init__()
        self.pe = nn.Embedding(2, d_model)

    def forward(self, x):
        B, N, L, _K = x.shape
        idx = torch.ones((B, N, L), device=x.device).long()
        idx[:, 0, :] = 0
        x = x + self.pe(idx)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4}]
