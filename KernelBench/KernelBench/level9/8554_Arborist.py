import torch
import torch.nn as nn


class Arborist(nn.Module):

    def __init__(self, l_dim, r_dim, k=5):
        super(Arborist, self).__init__()
        self.u = nn.Linear(l_dim * 2, k, bias=False)
        self.W = nn.Bilinear(l_dim * 2, r_dim, k, bias=False)

    def forward(self, e1, e2, q):
        """
        e1: tensor of size (*, l_dim)
        e2: tensor of size (*, r_dim)

        return: tensor of size (*, 1)
        """
        e = torch.cat((e1, e2), -1)
        u = self.u(e)
        w = self.W(e, q)
        return torch.sum(u * w, dim=-1, keepdim=True)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'l_dim': 4, 'r_dim': 4}]
