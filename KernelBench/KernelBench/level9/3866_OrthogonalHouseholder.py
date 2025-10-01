import math
import torch
import torch.nn as nn


class OrthogonalHouseholder(nn.Module):

    def __init__(self, sz, bias=True):
        super(OrthogonalHouseholder, self).__init__()
        self.sz = sz
        self.bias = bias
        self.A = nn.Parameter(torch.empty((sz, sz)))
        self.b = nn.Parameter(torch.empty(sz)) if bias else 0.0
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.A.normal_(0, math.sqrt(2 / self.sz))
            if self.bias:
                self.b.fill_(0.0)

    def forward(self, x):
        norms_sq = torch.einsum('ij,ij->i', self.A, self.A)
        for i in range(self.sz):
            x = x - 2 * self.A[i] * (x @ self.A[i].unsqueeze(1)) / norms_sq[i]
        return x + self.b


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'sz': 4}]
