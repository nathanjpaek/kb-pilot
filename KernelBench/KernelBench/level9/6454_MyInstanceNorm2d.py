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


class MyInstanceNorm2d(nn.Module):

    def __init__(self, num_features, momentum=0.9, eps=1e-05, affine=False,
        track_running_stats=False):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        if affine:
            self.affine = AffineChannelwise(num_features)
        else:
            self.affine = None
        self.track_running_stats = track_running_stats
        if track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

    def forward(self, x):
        assert len(x.shape) == 4
        b, c, h, w = x.shape
        if self.training or not self.track_running_stats:
            mu = x.mean(dim=(2, 3))
            sigma = x.var(dim=(2, 3), unbiased=False)
        else:
            mu, sigma = self.running_mean, self.running_var
            b = 1
        if self.training and self.track_running_stats:
            sigma_unbiased = sigma * (h * w / (h * w - 1))
            self.running_mean = self.running_mean * (1 - self.momentum
                ) + mu.mean(dim=0) * self.momentum
            self.running_var = self.running_var * (1 - self.momentum
                ) + sigma_unbiased.mean(dim=0) * self.momentum
        mu = mu.reshape(b, c, 1, 1)
        sigma = sigma.reshape(b, c, 1, 1)
        result = (x - mu) / torch.sqrt(sigma + self.eps)
        if self.affine is not None:
            result = self.affine(result)
        return result


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_features': 4}]
