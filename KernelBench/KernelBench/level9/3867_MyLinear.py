import torch
import torch.nn as nn


class MyLinear(nn.Module):

    def __init__(self, in_sz, out_sz, bias=True):
        super(MyLinear, self).__init__()
        self.in_sz = in_sz
        self.out_sz = out_sz
        self.bias = bias
        self.W = nn.Parameter(torch.empty((in_sz, out_sz)))
        self.b = nn.Parameter(torch.empty(1, out_sz)) if bias else 0.0
        self.reset_parameters()

    def forward(self, x):
        return x @ self.W + self.b

    def regularization(self):
        if self.in_sz == self.out_sz:
            return 2 * ((self.W @ self.W.T - torch.eye(self.in_sz, device=
                self.W.device)) ** 2).sum()
        return ((self.W @ self.W.T - torch.eye(self.in_sz, device=self.W.
            device)) ** 2).sum() + ((self.W.T @ self.W - torch.eye(self.
            out_sz, device=self.W.device)) ** 2).sum()

    def reset_parameters(self):
        with torch.no_grad():
            torch.nn.init.orthogonal_(self.W)
            if self.bias:
                self.b.fill_(0.0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_sz': 4, 'out_sz': 4}]
