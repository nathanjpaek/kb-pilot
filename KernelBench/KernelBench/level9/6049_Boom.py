import torch
import torch.nn as nn


class Boom(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1, shortcut
        =False, output_size=512):
        super(Boom, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout) if dropout else None
        if not shortcut:
            self.linear2 = nn.Linear(dim_feedforward, output_size)
        self.shortcut = shortcut
        self.act = nn.GELU()

    def forward(self, input):
        x = self.act(self.linear1(input))
        if self.dropout:
            x = self.dropout(x)
        if self.shortcut:
            ninp = input.shape[-1]
            x = torch.narrow(x, -1, 0, x.shape[-1] // ninp * ninp)
            x = x.view(*x.shape[:-1], x.shape[-1] // ninp, ninp)
            z = x.sum(dim=-2)
        else:
            z = self.linear2(x)
        return z


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4}]
