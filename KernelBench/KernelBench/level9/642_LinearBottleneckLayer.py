import torch
import torch.nn as nn


class LinearBottleneckLayer(nn.Module):
    """ Bottleneck Layer """

    def __init__(self, d_features, d_hid, d_out=None, dropout=0.1):
        super().__init__()
        if d_out is None:
            d_out = d_features
        self.encode = nn.Linear(d_features, d_hid)
        self.decode = nn.Linear(d_hid, d_out)
        nn.init.xavier_normal_(self.encode.weight)
        nn.init.xavier_normal_(self.decode.weight)
        self.layer_norm = nn.LayerNorm(d_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
            Arguments:
                x {Tensor, shape [batch_size, d_features]}
            Returns:
                x {Tensor, shape [batch_size, d_features]}
        """
        residual = x
        encode = nn.functional.relu(self.encode(x))
        decode = self.decode(encode)
        output = self.dropout(decode)
        output = self.layer_norm(output + residual)
        output = output + residual
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_features': 4, 'd_hid': 4}]
