import torch
import numpy as np
import torch.nn as nn


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_x, d_k, d_v):
        super().__init__()
        self.d_x = d_x
        self.d_k = d_k
        self.d_v = d_v
        self.w_q = nn.Linear(d_x, d_k)
        self.w_k = nn.Linear(d_x, d_k)
        self.w_v = nn.Linear(d_x, d_v)

    def forward(self, x):
        self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        logits = torch.bmm(K, V.permute(0, 2, 1)) / np.sqrt(self.d_k)
        return torch.bmm(torch.softmax(logits, dim=-1), V)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'d_x': 4, 'd_k': 4, 'd_v': 4}]
