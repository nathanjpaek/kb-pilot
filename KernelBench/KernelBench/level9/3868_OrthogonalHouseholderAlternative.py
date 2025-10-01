import math
import torch
import torch.nn as nn


class OrthogonalHouseholderAlternative(nn.Module):

    def __init__(self, sz, bias=True):
        super(OrthogonalHouseholderAlternative, self).__init__()
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

    def _forward_precalc(self):
        B = self.A @ self.A.T
        self.diag = torch.diag(B)
        self.p = self.A.clone()
        for i in range(self.sz - 1):
            self.p[i + 1:] = self.p[i + 1:].clone() - (2 * B[i, i + 1:] /
                self.diag[i + 1:]).unsqueeze(1) * self.p[i].clone()

    def forward(self, x):
        self._forward_precalc()
        B = x @ self.A.T
        x = x - 2 * B / self.diag @ self.p
        return x + self.b


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'sz': 4}]
