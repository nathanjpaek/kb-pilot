import math
import torch
import torch.nn as nn


def compute_negative_ln_prob(Y, mu, ln_var, pdf):
    var = ln_var.exp()
    if pdf == 'gauss':
        negative_ln_prob = 0.5 * ((Y - mu) ** 2 / var).sum(1).mean(
            ) + 0.5 * Y.size(1) * math.log(2 * math.pi) + 0.5 * ln_var.sum(1
            ).mean()
    elif pdf == 'logistic':
        whitened = (Y - mu) / var
        adjust = torch.logsumexp(torch.stack([torch.zeros(Y.size()), -
            whitened]), 0)
        negative_ln_prob = whitened.sum(1).mean() + 2 * adjust.sum(1).mean(
            ) + ln_var.sum(1).mean()
    else:
        raise ValueError('Unknown PDF: %s' % pdf)
    return negative_ln_prob


class PDF(nn.Module):

    def __init__(self, dim, pdf):
        super(PDF, self).__init__()
        assert pdf in {'gauss', 'logistic'}
        self.dim = dim
        self.pdf = pdf
        self.mu = nn.Embedding(1, self.dim)
        self.ln_var = nn.Embedding(1, self.dim)

    def forward(self, Y):
        cross_entropy = compute_negative_ln_prob(Y, self.mu.weight, self.
            ln_var.weight, self.pdf)
        return cross_entropy


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4, 'pdf': 'gauss'}]
