import torch
import torch.nn as nn


class LayerNormCustom(nn.Module):
    """A layernorm module in the TF style (epsilon inside the square root)."""

    def __init__(self, n_hidden, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(n_hidden))
        self.beta = nn.Parameter(torch.zeros(n_hidden))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_hidden': 4}]
