import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.functional as F


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.onelayer = d_hid == d_in
        if self.onelayer:
            self.w = nn.Linear(d_in, d_in, bias=False)
            self.tanh = nn.Tanh()
        else:
            self.w_1 = nn.Conv1d(d_in, d_hid, 1)
            self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        if self.onelayer:
            output = self.w(x)
            output = self.tanh(output)
        else:
            output = x.transpose(1, 2)
            output = self.w_2(F.relu(self.w_1(output)))
            output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_in': 4, 'd_hid': 4}]
