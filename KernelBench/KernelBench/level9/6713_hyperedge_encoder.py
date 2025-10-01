import torch
import torch.nn as nn
import torch.nn.functional as F


class hyperedge_encoder(nn.Module):

    def __init__(self, num_in_edge, num_hidden, dropout, act=F.tanh):
        super(hyperedge_encoder, self).__init__()
        self.num_in_edge = num_in_edge
        self.num_hidden = num_hidden
        self.dropout = dropout
        self.act = act
        self.W1 = nn.Parameter(torch.zeros(size=(self.num_in_edge, self.
            num_hidden), dtype=torch.float))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        self.b1 = nn.Parameter(torch.zeros(self.num_hidden, dtype=torch.float))

    def forward(self, H_T):
        z1 = self.act(torch.mm(H_T, self.W1) + self.b1)
        return z1

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_in_edge
            ) + ' -> ' + str(self.num_hidden)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'num_in_edge': 4, 'num_hidden': 4, 'dropout': 0.5}]
