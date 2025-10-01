import torch
from torch import nn


class FC(nn.Module):

    def __init__(self, n_dim_in, n_dim_out, equal_lr=True):
        super().__init__()
        norm_const = n_dim_in ** -0.5
        scale_init = 1 if equal_lr else norm_const
        self.scale_forward = norm_const if equal_lr else 1
        self.weight = nn.Parameter(scale_init * torch.randn(n_dim_out,
            n_dim_in))
        self.bias = nn.Parameter(torch.zeros(n_dim_out))

    def forward(self, x):
        return nn.functional.linear(x, self.scale_forward * self.weight,
            bias=self.bias)


class AffineTransform(nn.Module):

    def __init__(self, n_dim_w, n_feature_maps, equal_lr):
        super().__init__()
        self.fc = FC(n_dim_w, n_feature_maps, equal_lr=equal_lr)
        nn.init.ones_(self.fc.bias)

    def forward(self, w):
        return self.fc(w)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_dim_w': 4, 'n_feature_maps': 4, 'equal_lr': 4}]
