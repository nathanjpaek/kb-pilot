import torch
import torch.nn as nn


class LBM(nn.Module):

    def __init__(self, l_dim, r_dim):
        super(LBM, self).__init__()
        self.W = nn.Bilinear(l_dim, r_dim, 1, bias=False)

    def forward(self, e1, e2):
        """
        e1: tensor of size (*, l_dim)
        e2: tensor of size (*, r_dim)

        return: tensor of size (*, 1)
        """
        return torch.exp(self.W(e1, e2))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'l_dim': 4, 'r_dim': 4}]
