import torch
import torch.nn as nn


class LgRegv(torch.nn.Module):
    """
    TODO: pre-training
    from power to voronoi
    """

    def __init__(self, dim, nla):
        super(LgRegv, self).__init__()
        self.linear = nn.Linear(dim, nla, bias=False)

    def forward(self, x):
        ba = -torch.sum((self.linear.weight / 2) ** 2, dim=1)
        y_hat = self.linear(x) + ba
        y_hat = torch.sigmoid(y_hat)
        return y_hat


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4, 'nla': 4}]
