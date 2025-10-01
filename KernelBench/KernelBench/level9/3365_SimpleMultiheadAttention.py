import torch
from torch import nn


class SimpleMultiheadAttention(nn.Module):

    def __init__(self, d_x, d_attn, num_heads):
        super(SimpleMultiheadAttention, self).__init__()
        self.single_head_attn = nn.Linear(d_x, d_attn)
        self.multi_head_attn = nn.Linear(d_attn, num_heads)

    def forward(self, x):
        y = self.single_head_attn(x)
        nn.functional.relu(y)
        y = self.multi_head_attn(y)
        y = nn.functional.softmax(y)
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_x': 4, 'd_attn': 4, 'num_heads': 4}]
