import torch
import torch.nn as nn


class Linear(nn.Linear):
    """
    Apply linear projection to the last dimention of a tensor.
    """

    def forward(self, x):
        size = x.size()
        return super().forward(x.contiguous().view(-1, size[-1])).view(*
            size[:-1], -1)


class ConcatAndProject(nn.Module):

    def __init__(self, input_dim, output_dim, dropout, activation=None,
        bias=True):
        super().__init__()
        self.input_dropout = nn.Dropout(dropout)
        self.linear1 = Linear(input_dim, output_dim, bias=bias)
        self.activation = activation

    def forward(self, *args):
        input = self.input_dropout(torch.cat(args, dim=-1))
        if self.activation is None:
            return self.linear1(input)
        else:
            return getattr(torch, self.activation)(self.linear1(input))


class PointerSwitch(nn.Module):

    def __init__(self, query_dim, key_dim, input_dropout):
        super().__init__()
        self.project = ConcatAndProject(query_dim + key_dim, 1,
            input_dropout, activation=None)

    def forward(self, query, key):
        return torch.sigmoid(self.project(query, key))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'query_dim': 4, 'key_dim': 4, 'input_dropout': 0.5}]
