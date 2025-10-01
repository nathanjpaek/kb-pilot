import torch
import torch.nn as nn


class RawArborist(nn.Module):

    def __init__(self, l_dim, r_dim, k=5):
        super(RawArborist, self).__init__()
        self.u = nn.Linear(l_dim, k, bias=False)
        self.W = nn.Bilinear(l_dim, r_dim, k, bias=False)

    def forward(self, e, q):
        u = self.u(e)
        w = self.W(e, q)
        return torch.sum(u * w, dim=-1, keepdim=True)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'l_dim': 4, 'r_dim': 4}]
