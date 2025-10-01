import torch
import torch.nn as nn


class PCENlr(nn.Module):
    """
        A Low-rank version for per-channel energy normalization.
    """

    def __init__(self, N, T):
        super(PCENlr, self).__init__()
        self.N = N
        self.T = T
        self.lr_enc = nn.Linear(self.T, 1, bias=False)
        self.lr_dec = nn.Linear(1, self.T, bias=False)
        self.log_alpha = nn.Parameter((torch.randn(self.N) * 0.1 + 1.0).log_())
        self.log_delta = nn.Parameter((torch.randn(self.N) * 0.1 + 2.0).log_())
        self.log_rho = nn.Parameter((torch.randn(self.N) * 0.1 + 0.6).log_())
        self.relu = nn.ReLU()
        self.eps = 0.1
        self.initialize_parameters()

    def initialize_parameters(self):
        torch.nn.init.xavier_normal(self.lr_enc.weight)

    def forward(self, x):
        alpha = self.log_alpha.expand_as(x).exp()
        delta = self.log_delta.expand_as(x).exp()
        rho = self.log_rho.expand_as(x).exp()
        m = self.relu(self.lr_dec(self.lr_enc(x.permute(0, 2, 1))).permute(
            0, 2, 1) + x)
        pcen_out = (x / (m + self.eps) ** alpha + delta) ** rho - delta ** rho
        return pcen_out


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'N': 4, 'T': 4}]
