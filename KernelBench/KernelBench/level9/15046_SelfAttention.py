import torch
import torch.nn as nn


class SelfAttention(nn.Module):

    def __init__(self, *args, **kargs):
        super().__init__()
        self.attention = nn.MultiheadAttention(*args, **kargs)

    def forward(self, x):
        return self.attention(x, x, x)[0]


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'embed_dim': 4, 'num_heads': 4}]
