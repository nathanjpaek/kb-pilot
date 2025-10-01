import torch
import torch.utils.data
import torch.nn as nn


class LayerNorm(nn.Module):

    def __init__(self, d):
        super().__init__()
        self.a = nn.Parameter(torch.ones(d).unsqueeze(0).unsqueeze(0))
        self.b = nn.Parameter(torch.zeros(d).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        mean = x.mean(dim=(0, 1), keepdim=True)
        var = x.var(dim=(0, 1), keepdim=True, unbiased=False)
        x = self.a * (x - mean) / torch.sqrt(var + 1e-06) + self.b
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d': 4}]
