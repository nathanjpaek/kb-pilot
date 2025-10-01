import torch
from torch import nn


class GATgate_lp(nn.Module):

    def __init__(self, n_dim):
        super(GATgate_lp, self).__init__()
        self.w_l1 = nn.Linear(n_dim, n_dim)
        self.w_l2 = nn.Linear(n_dim, n_dim)
        self.w_p1 = nn.Linear(n_dim, n_dim)
        self.w_p2 = nn.Linear(n_dim, n_dim)
        self.LR = nn.LeakyReLU()

    def forward(self, vec_l, vec_p, adj_inter):
        h_l = self.w_l1(vec_l)
        h_p = self.w_p1(vec_p)
        h_l2 = torch.einsum('aij,ajk->aik', (adj_inter, h_p))
        h_l2 = self.LR(self.w_l2(h_l2 * h_l))
        h_p2 = torch.einsum('aij,ajk->aik', (adj_inter.transpose(-1, -2), h_l))
        h_p2 = self.LR(self.w_p2(h_p2 * h_p))
        return h_l2, h_p2


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'n_dim': 4}]
