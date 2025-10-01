import torch
from torch import Tensor
from torch import nn


class MLP(nn.Module):

    def __init__(self, dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(dim, embed_dim)

    def forward(self, x: 'Tensor') ->Tensor:
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4, 'embed_dim': 4}]
