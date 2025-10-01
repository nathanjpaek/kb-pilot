import torch
from torch import nn
import torch.nn.functional as F


class LocalMLP(nn.Module):

    def __init__(self, dim_in: 'int', use_norm: 'bool'=True):
        """a Local 1 layer MLP

        :param dim_in: feat in size
        :type dim_in: int
        :param use_norm: if to apply layer norm, defaults to True
        :type use_norm: bool, optional
        """
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_in, bias=not use_norm)
        self.use_norm = use_norm
        if use_norm:
            self.norm = nn.LayerNorm(dim_in)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """forward of the module

        :param x: input tensor (..., dim_in)
        :type x: torch.Tensor
        :return: output tensor (..., dim_in)
        :rtype: torch.Tensor
        """
        x = self.linear(x)
        if hasattr(self, 'norm'):
            x = self.norm(x)
        x = F.relu(x, inplace=True)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_in': 4}]
