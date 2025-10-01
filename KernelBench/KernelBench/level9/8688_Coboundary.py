import torch
import torch.nn as nn
import torch.nn.functional


class Coboundary(nn.Module):

    def __init__(self, C_in, C_out, enable_bias=True, variance=1.0):
        super().__init__()
        assert C_in > 0
        assert C_out > 0
        self.C_in = C_in
        self.C_out = C_out
        self.enable_bias = enable_bias
        self.theta = nn.parameter.Parameter(variance * torch.randn((self.
            C_out, self.C_in)))
        if self.enable_bias:
            self.bias = nn.parameter.Parameter(torch.zeros((1, self.C_out, 1)))
        else:
            self.bias = 0.0

    def forward(self, D, x):
        assert len(D.shape) == 2
        B, C_in, M = x.shape
        assert D.shape[1] == M
        assert C_in == self.C_in
        N = D.shape[0]
        X = []
        for b in range(0, B):
            X12 = []
            for c_in in range(0, self.C_in):
                X12.append(D.mm(x[b, c_in, :].unsqueeze(1)).transpose(0, 1))
            X12 = torch.cat(X12, 0)
            assert X12.shape == (self.C_in, N)
            X.append(X12.unsqueeze(0))
        X = torch.cat(X, 0)
        assert X.shape == (B, self.C_in, N)
        y = torch.einsum('oi,bin->bon', (self.theta, X))
        assert y.shape == (B, self.C_out, N)
        return y + self.bias


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'C_in': 4, 'C_out': 4}]
