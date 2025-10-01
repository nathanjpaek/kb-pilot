import torch
import torch.nn as nn


class Norm(nn.Module):
    """
    Re-usable class for either batch-norm or layer-norm (by swapping dim)
    """

    def __init__(self, n_hidden, eps=1e-08, dim=0):
        super(Norm, self).__init__()
        self.eps = eps
        self.n_hidden = n_hidden
        self.a = nn.Parameter(torch.ones(1, n_hidden), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(1, n_hidden), requires_grad=True)
        self.dim = dim

    def forward(self, x):
        mean_x = torch.mean(x, dim=self.dim).expand_as(x)
        std_x = torch.std(x, dim=self.dim).expand_as(x)
        out = (x - mean_x) / (std_x + self.eps)
        out = out * self.a.expand_as(x) + self.b.expand_as(x)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_hidden': 4}]
