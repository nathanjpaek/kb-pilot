import torch
import torch.utils.data
from torch import nn


class NormalisedSigmoid(nn.Module):
    """ Normalised logistic sigmoid function. """

    def __init__(self, p: 'float'=1, dim: 'int'=-1):
        super().__init__()
        self.p = p
        self.dim = dim

    def forward(self, s: 'torch.Tensor') ->torch.Tensor:
        a = torch.sigmoid(s)
        return torch.nn.functional.normalize(a, p=self.p, dim=self.dim)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
