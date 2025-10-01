import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Norm to 0-mean 1-std , then do a learned diagonal affine transform."""

    def __init__(self, features, eps=1e-05):
        super(LayerNorm, self).__init__()
        self.scale = nn.Parameter(torch.ones(features))
        self.shift = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        s = (x - mean).pow(2).mean(-1, keepdim=True)
        x = (x - mean) * torch.rsqrt(s + self.eps)
        return self.scale * x + self.shift


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'features': 4}]
