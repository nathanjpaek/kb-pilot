import torch
from torch import nn


class AffineChannelwise(nn.Module):

    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        self.register_parameter('weight', nn.Parameter(torch.ones(
            num_channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(num_channels))
            )

    def forward(self, x):
        param_shape = [1] * len(x.shape)
        param_shape[1] = self.num_channels
        return x * self.weight.reshape(*param_shape) + self.bias.reshape(*
            param_shape)


class MyGroupNorm(nn.Module):

    def __init__(self, num_groups, num_channels, eps=1e-05, affine=True):
        super().__init__()
        assert num_channels % num_groups == 0
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        if affine:
            self.affine = AffineChannelwise(num_channels)
        else:
            self.affine = None

    def forward(self, x):
        assert len(x.shape) == 4
        b, c, h, w = x.shape
        assert c == self.num_channels
        g = c // self.num_groups
        x = x.reshape(b, self.num_groups, g, h, w)
        mu = x.mean(dim=(2, 3, 4), keepdim=True)
        sigma = x.var(dim=(2, 3, 4), unbiased=False, keepdim=True)
        result = (x - mu) / torch.sqrt(sigma + self.eps)
        result = result.reshape(b, c, h, w)
        if self.affine is not None:
            result = self.affine(result)
        return result


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_groups': 1, 'num_channels': 4}]
