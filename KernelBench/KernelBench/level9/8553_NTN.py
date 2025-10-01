import torch
import torch.nn as nn
import torch.nn.functional as F


class NTN(nn.Module):

    def __init__(self, l_dim, r_dim, k=5, non_linear=F.tanh):
        super(NTN, self).__init__()
        self.u_R = nn.Linear(k, 1, bias=False)
        self.f = non_linear
        self.W = nn.Bilinear(l_dim * 2, r_dim, k, bias=False)
        self.V = nn.Linear(l_dim * 2 + r_dim, k, bias=False)

    def forward(self, e1, e2, q):
        """
        e1: tensor of size (*, l_dim)
        e2: tensor of size (*, r_dim)

        return: tensor of size (*, 1)
        """
        e = torch.cat((e1, e2), -1)
        return self.u_R(self.f(self.W(e, q) + self.V(torch.cat((e, q), 1))))


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'l_dim': 4, 'r_dim': 4}]
