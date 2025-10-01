import torch
import torch.nn as nn


class GatedLinearUnit(nn.Module):

    def __init__(self, dim: 'int'=-1, nonlinear: 'bool'=True):
        super().__init__()
        self.dim = dim
        self.nonlinear = nonlinear

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        val, gate = torch.chunk(x, 2, dim=self.dim)
        if self.nonlinear:
            val = torch.tanh(val)
        return torch.sigmoid(gate) * val


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
