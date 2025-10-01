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


class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff=2048, dropout=0.1, activation='relu'):
        super(FeedForward, self).__init__()
        self.linear1 = Linear(d_model, d_ff, w_init=activation)
        self.linear2 = Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        output = self.linear2(F.relu(self.linear1(x)))
        output = self.dropout(output)
        output = x + output
        output = self.norm(output)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4}]
