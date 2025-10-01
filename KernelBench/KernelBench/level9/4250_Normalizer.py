import torch
import torch.nn as nn


class Normalizer(nn.Module):

    def __init__(self, dim=-1, norm=1.0):
        super().__init__()
        self.dim = dim
        self.norm = norm
        self.softplus = nn.Softplus()

    def forward(self, x):
        out = self.softplus(x)
        return out / torch.abs(out.detach()).sum(dim=self.dim, keepdims=True)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
