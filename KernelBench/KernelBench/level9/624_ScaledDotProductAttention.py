import torch
import numpy as np
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, query, key, value, mask=None):
        """
            Arguments:
                query {Tensor, shape [n_head * batch, q_length, dk]} -- query
                key {Tensor, shape [n_head * batch, k_length, dk]} -- key
                value {Tensor, shape [n_head * batch, v_length, dv]} -- value
                mask {Tensor, shape [n_head * batch, q_length, k_length]} --self attn mask

            Returns:
                output {Tensor, shape [n_head * batch, q_length, dv] -- output
                attn {Tensor, shape [n_head * batch, q_length, k_length] -- self attention

        """
        attn = torch.bmm(query, key.transpose(1, 2))
        attn = attn / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, value)
        return output, attn


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'temperature': 4}]
