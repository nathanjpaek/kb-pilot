import torch
from torch import nn


class LayerNorm(nn.Module):

    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.Tensor(num_channels))
        self.bias = nn.Parameter(torch.Tensor(num_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        mu = torch.mean(x, axis=(1, 2), keepdims=True)
        sig = torch.sqrt(torch.mean((x - mu) ** 2, axis=(1, 2), keepdims=
            True) + self.eps)
        return (x - mu) / sig * self.weight[:, None, None] + self.bias[:,
            None, None]


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_channels': 4}]
