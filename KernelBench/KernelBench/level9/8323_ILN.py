import torch
import torch.utils.data
import torch.utils.data.distributed
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class ILN(nn.Module):

    def __init__(self, num_features, eps=1e-05):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, x):
        in_mean, in_var = torch.mean(x, dim=[2, 3], keepdim=True), torch.var(x,
            dim=[2, 3], keepdim=True)
        out_in = (x - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(x, dim=[1, 2, 3], keepdim=True
            ), torch.var(x, dim=[1, 2, 3], keepdim=True)
        out_ln = (x - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(x.shape[0], -1, -1, -1) * out_in + (1 - self.
            rho.expand(x.shape[0], -1, -1, -1)) * out_ln
        out = out * self.gamma.expand(x.shape[0], -1, -1, -1
            ) + self.beta.expand(x.shape[0], -1, -1, -1)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_features': 4}]
