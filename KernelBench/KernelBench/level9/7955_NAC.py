import torch
from torch import nn


class NAC(nn.Module):

    def __init__(self, in_dim, out_dim, init_fun=nn.init.xavier_uniform_):
        super().__init__()
        self._W_hat = nn.Parameter(torch.empty(in_dim, out_dim))
        self._M_hat = nn.Parameter(torch.empty(in_dim, out_dim))
        self.register_parameter('W_hat', self._W_hat)
        self.register_parameter('M_hat', self._M_hat)
        for param in self.parameters():
            init_fun(param)

    def forward(self, x):
        W = torch.tanh(self._W_hat) * torch.sigmoid(self._M_hat)
        return x.matmul(W)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4}]
