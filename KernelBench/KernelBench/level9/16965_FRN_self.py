import torch
import torch.nn as nn


class FRN_self(nn.Module):

    def __init__(self, num_features, eps=1e-05, is_eps_learnable=True):
        super(FRN_self, self).__init__()
        self.num_features = num_features
        self.init_eps = eps
        self.is_eps_learnable = is_eps_learnable
        self.gamma = nn.Parameter(torch.Tensor(1, num_features, 1, 1),
            requires_grad=True)
        self.beta = nn.Parameter(torch.Tensor(1, num_features, 1, 1),
            requires_grad=True)
        self.eps = nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        if self.is_eps_learnable:
            nn.init.constant_(self.eps, self.init_eps)

    def extra_repr(self):
        return 'num_features={num_features}, eps={init_eps}'.format(**self.
            __dict__)

    def forward(self, x):
        nu2 = x.pow(2).mean(dim=[2, 3], keepdim=True)
        x = x * torch.rsqrt(nu2 + self.eps.abs())
        return self.gamma * x + self.beta


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_features': 4}]
