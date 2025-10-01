import torch
import torch.nn as nn


class ReactionDotProduction(nn.Module):
    """ Scaled Dot Productionss """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, query, key, value):
        """
            Arguments:
                key {Tensor, shape [n_head * batch, d_features, n_depth_per_head]} -- expansion
                query {Tensor, shape [n_head * batch, 1, n_depth_per_head]} -- depth
                value {Tensor, shape [n_head * batch, 1, d_features]} -- value

            Returns:
                output {Tensor, shape [n_head * batch, 1, d_features]} -- output
                attn {Tensor, shape [n_head * batch, 1, d_features]} -- reaction attention
        """
        attn = torch.bmm(query, key.transpose(1, 2))
        attn = attn / self.temperature
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.mul(attn, value)
        return output, attn


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'temperature': 4}]
