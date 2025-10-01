import torch
import torch.utils.data
from torch import nn


class PerfectProd(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()

    def reset_parameters(self):
        pass

    def forward(self, x):
        return torch.prod(2 * x[:, :-1], dim=-1, keepdim=True)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
