import torch
import torch.nn.functional as F
import torch.nn as nn


class Linear(nn.Module):

    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.
            calculate_gain(w_init))

    def forward(self, x):
        return self.linear(x)


class Prenet(nn.Module):
    """Some Information about Prenet"""

    def __init__(self, n_mel_channels, d_model):
        super(Prenet, self).__init__()
        self.linear1 = Linear(n_mel_channels, d_model, bias=False)
        self.linear2 = Linear(d_model, d_model, bias=False)

    def forward(self, x):
        x = F.dropout(F.relu(self.linear1(x)), p=0.5, training=True)
        x = F.dropout(F.relu(self.linear2(x)), p=0.5, training=True)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_mel_channels': 4, 'd_model': 4}]
