import math
import torch
import numpy as np
import torch.nn as nn


class NotearsSobolev(nn.Module):

    def __init__(self, d, k):
        """d: num variables k: num expansion of each variable"""
        super(NotearsSobolev, self).__init__()
        self.d, self.k = d, k
        self.fc1_pos = nn.Linear(d * k, d, bias=False)
        self.fc1_neg = nn.Linear(d * k, d, bias=False)
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        nn.init.zeros_(self.fc1_pos.weight)
        nn.init.zeros_(self.fc1_neg.weight)
        self.l2_reg_store = None

    def _bounds(self):
        bounds = []
        for j in range(self.d):
            for i in range(self.d):
                for _ in range(self.k):
                    if i == j:
                        bound = 0, 0
                    else:
                        bound = 0, None
                    bounds.append(bound)
        return bounds

    def sobolev_basis(self, x):
        seq = []
        for kk in range(self.k):
            mu = 2.0 / (2 * kk + 1) / math.pi
            psi = mu * torch.sin(x / mu)
            seq.append(psi)
        bases = torch.stack(seq, dim=2)
        bases = bases.view(-1, self.d * self.k)
        return bases

    def forward(self, x):
        bases = self.sobolev_basis(x)
        x = self.fc1_pos(bases) - self.fc1_neg(bases)
        self.l2_reg_store = torch.sum(x ** 2) / x.shape[0]
        return x

    def h_func(self):
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight
        fc1_weight = fc1_weight.view(self.d, self.d, self.k)
        A = torch.sum(fc1_weight * fc1_weight, dim=2).t()
        h = trace_expm(A) - d
        return h

    def l2_reg(self):
        reg = self.l2_reg_store
        return reg

    def fc1_l1_reg(self):
        reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        return reg

    @torch.no_grad()
    def fc1_to_adj(self) ->np.ndarray:
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight
        fc1_weight = fc1_weight.view(self.d, self.d, self.k)
        A = torch.sum(fc1_weight * fc1_weight, dim=2).t()
        W = torch.sqrt(A)
        W = W.cpu().detach().numpy()
        return W


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d': 4, 'k': 4}]
