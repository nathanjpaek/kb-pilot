import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-06)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
            Arguments:
                x {Tensor, shape [batch_size, length, d_features]}

            Returns:
                x {Tensor, shape [batch_size, length, d_features]}

        """
        residual = x
        x = self.layer_norm(x)
        x = self.w_1(x)
        x = nn.functional.relu(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x += residual
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_in': 4, 'd_hid': 4}]
