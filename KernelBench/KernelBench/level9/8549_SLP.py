import torch
import torch.nn as nn
import torch.nn.functional as F


class SLP(nn.Module):

    def __init__(self, l_dim, r_dim, hidden_dim, non_linear=F.tanh):
        super(SLP, self).__init__()
        self.u_R = nn.Linear(hidden_dim, 1, bias=False)
        self.f = non_linear
        self.ffn = nn.Linear(l_dim * 2 + r_dim, hidden_dim, bias=False)

    def forward(self, e1, e2, q):
        """
        e1: tensor of size (*, l_dim)
        e2: tensor of size (*, r_dim)

        return: tensor of size (*, 1)
        """
        return self.u_R(self.f(self.ffn(torch.cat((e1, e2, q), 1))))


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'l_dim': 4, 'r_dim': 4, 'hidden_dim': 4}]
