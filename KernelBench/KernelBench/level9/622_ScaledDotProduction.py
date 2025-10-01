import torch
import torch.nn as nn


class ScaledDotProduction(nn.Module):
    """Scaled Dot Production"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, query, key, value):
        """
            Arguments:
                query {Tensor, shape: [batch, d_k, d_out]} -- query
                key {Tensor, shape: [batch, d_k, n_candidate]} -- key
                value {Tensor, shape: [batch, d_v, n_candidate]} -- value

            Returns:
                output {Tensor, shape [n_head * batch, n_depth, n_vchannel * d_features] -- output
                attn {Tensor, shape [n_head * batch, n_depth, n_depth] -- reaction attention
        """
        attn = torch.bmm(query.transpose(2, 1), key)
        attn = attn / self.temperature
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, value.transpose(2, 1))
        return output, attn


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'temperature': 4}]
