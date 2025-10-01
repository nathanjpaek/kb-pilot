import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_kl(log_alpha):
    return 0.5 * torch.sum(torch.log1p(torch.exp(-log_alpha)))


class VdLinear(nn.Module):
    """
    Linear Layer variational dropout

    """

    def __init__(self, n_in, n_out, alpha_shape=(1, 1), bias=True):
        super(VdLinear, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.alpha_shape = alpha_shape
        self.bias = bias
        self.W = nn.Parameter(torch.Tensor(self.n_out, self.n_in))
        self.log_alpha = nn.Parameter(torch.Tensor(*self.alpha_shape))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, self.n_out))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.kl_value = calculate_kl

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        self.log_alpha.data.fill_(-5.0)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, X, sample=False):
        mean = F.linear(X, self.W)
        if self.bias is not None:
            mean = mean + self.bias
        sigma = torch.exp(self.log_alpha) * self.W * self.W
        std = torch.sqrt(1e-16 + F.linear(X * X, sigma))
        if self.training or sample:
            epsilon = std.data.new(std.size()).normal_()
        else:
            epsilon = 0.0
        out = mean + std * epsilon
        kl = self.kl_loss()
        return out, kl

    def kl_loss(self):
        return self.W.nelement() * self.kl_value(self.log_alpha
            ) / self.log_alpha.nelement()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_in': 4, 'n_out': 4}]
