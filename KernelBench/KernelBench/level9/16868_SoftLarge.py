import math
import torch
from torch import nn


class SoftCompare(nn.Module):

    def __init__(self, alpha=None, beta=None):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * (0 if alpha is None else
            alpha), requires_grad=True)
        self.beta = nn.Parameter(torch.ones(1) * (0 if beta is None else
            beta), requires_grad=True)
        if alpha is None:
            nn.init.normal_(self.alpha.data, 0, 1)
        else:
            self.alpha.requires_grad_(False)
        if beta is not None:
            self.beta.requires_grad_(False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        raise NotImplementedError


class SoftLarge(SoftCompare):
    """
    Sigmoid((x - alpha) / e^beta)
    """

    def __init__(self, alpha=None, beta=None):
        super().__init__(alpha, beta)

    def forward(self, x, beta=None):
        alpha = self.alpha
        if beta is None:
            beta = self.beta
        return self.sigmoid((x - alpha) / torch.exp(beta))

    def show(self, name='SoftLarge', indent=0, log=print, **kwargs):
        alpha = kwargs['alpha'] if 'alpha' in kwargs else self.alpha
        beta = kwargs['beta'] if 'beta' in kwargs else self.beta
        log(' ' * indent + '- %s(x) = Sigmoid((x - %lf) / %lf)' % (name,
            alpha, math.exp(beta)))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
