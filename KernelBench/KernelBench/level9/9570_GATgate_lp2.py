import torch
from torch import nn


class GATgate_lp2(nn.Module):

    def __init__(self, n_dim):
        super(GATgate_lp2, self).__init__()
        self.w_l = nn.Linear(n_dim, n_dim)
        self.w_p = nn.Linear(n_dim, n_dim)
        self.LR = nn.LeakyReLU()

    def forward(self, vec_l, vec_p, adj_inter):
        h_l = self.w_l(vec_l)
        h_p = self.w_p(vec_p)
        intermat = torch.einsum('aij,ajk->aik', (h_l, h_p.transpose(-1, -2)))
        intermat = intermat * adj_inter
        return intermat


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'n_dim': 4}]
