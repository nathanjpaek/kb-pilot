import torch
from torch import nn


class TransformerSet(nn.Module):

    def __init__(self, input_size, dropout=0.5, trans_head_nums=1, **kwargs):
        super(TransformerSet, self).__init__()
        self.Transformer = nn.MultiheadAttention(embed_dim=input_size,
            num_heads=trans_head_nums, dropout=dropout)
        self.fc = nn.Linear(input_size, input_size)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(input_size)

    def forward(self, x, lens=None):
        x = x.transpose(0, 1).contiguous()
        residual, _weights = self.Transformer(x, x, x)
        residual = self.dropout(self.fc(residual))
        return self.layernorm(residual + x).transpose(0, 1).contiguous()


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4}]
