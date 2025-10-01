import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, input_size, num_heads, drop_prob=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.drop_prob = drop_prob
        self.multihead_attention = nn.MultiheadAttention(input_size, num_heads)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        attn_output, _attn_output_weights = self.multihead_attention(x, x, x)
        return attn_output


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'num_heads': 4}]
