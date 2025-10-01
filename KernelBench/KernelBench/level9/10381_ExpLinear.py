import torch
from torch import nn
import torch.nn
from scipy.linalg import logm


class InverseNotAvailable(Exception):
    """Exception to be thrown when a transform does not have an inverse."""
    pass


class Transform(nn.Module):
    """Base class for all transform objects."""

    def forward(self, inputs, context=None):
        raise NotImplementedError()

    def inverse(self, inputs, context=None):
        raise InverseNotAvailable()


class ExpLinear(Transform):

    def __init__(self, d):
        super().__init__()
        self.d = d
        dummy = nn.Linear(d, d)
        self.A = nn.Parameter(torch.tensor(logm(dummy.weight.detach().numpy
            ()), dtype=torch.float32))
        self.b = nn.Parameter(dummy.bias.detach())

    def forward(self, x, context=None):
        W = torch.matrix_exp(self.A)
        z = torch.matmul(x, W) + self.b
        logabsdet = torch.trace(self.A) * x.new_ones(z.shape[0])
        return z, logabsdet

    def inverse(self, z, context=None):
        W_inv = torch.matrix_exp(-self.A)
        x = torch.matmul(z - self.b, W_inv)
        logabsdet = -torch.trace(self.A) * z.new_ones(x.shape[0])
        return x, logabsdet


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d': 4}]
