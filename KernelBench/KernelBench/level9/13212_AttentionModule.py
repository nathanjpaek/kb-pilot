import torch
from torch import nn
import torch.nn.functional as F


class AttentionModule(nn.Module):

    def __init__(self, d_model, d_k=None, device='cpu', dropout=None):
        super().__init__()
        if not d_k:
            d_k = d_model
        self.W = nn.Parameter(torch.randn(d_model, d_model, device=device))
        self.bias = nn.Parameter(torch.randn(1, device=device))
        self.dropout = dropout
        self.norm = nn.LayerNorm(d_model)

    def forward(self, key, query, value):
        key = self.norm(key)
        query = self.norm(query)
        value = self.norm(value)
        query_W_key = torch.bmm(torch.matmul(query, self.W), key.transpose(
            -2, -1))
        if self.dropout:
            query_W_key = self.dropout(query_W_key)
        weights = F.softmax(torch.tanh(query_W_key + self.bias), dim=-1)
        return weights, torch.bmm(weights, value)


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'d_model': 4}]
