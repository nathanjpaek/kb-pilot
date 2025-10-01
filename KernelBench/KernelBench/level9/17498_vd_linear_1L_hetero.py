import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


def calculate_kl(log_alpha):
    return 0.5 * torch.sum(torch.log1p(torch.exp(-log_alpha)))


class VdLinear(nn.Module):
    """
    variational dropout

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


class vd_linear_1L_hetero(nn.Module):
    """1 hidden layer Variational Dropout Network"""

    def __init__(self, input_dim, output_dim, alpha_shape=(1, 1), bias=True,
        n_hid=50):
        super(vd_linear_1L_hetero, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha_shape = alpha_shape
        self.bias = bias
        self.bfc1 = VdLinear(input_dim, n_hid, self.alpha_shape, self.bias)
        self.bfc2 = VdLinear(n_hid, 2 * output_dim, self.alpha_shape, self.bias
            )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, sample=False):
        tkl = 0.0
        x = x.view(-1, self.input_dim)
        x, kl = self.bfc1(x, sample)
        tkl = tkl + kl
        x = self.act(x)
        y, kl = self.bfc2(x, sample)
        tkl = tkl + kl
        return y, tkl

    def sample_predict(self, x, Nsamples):
        """Used for estimating the data's likelihood by approximately marginalising the weights with MC"""
        predictions = x.data.new(Nsamples, x.shape[0], self.output_dim)
        tkl_vec = np.zeros(Nsamples)
        for i in range(Nsamples):
            y, tkl = self.forward(x, sample=True)
            predictions[i] = y
            tkl_vec[i] = tkl
        return predictions, tkl_vec


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4}]
