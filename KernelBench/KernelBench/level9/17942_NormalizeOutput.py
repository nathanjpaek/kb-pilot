import torch
import torch.nn.functional as F
from torch import nn
import torch.optim


class NormalizeOutput(nn.Module):
    """
    Module that scales the input tensor to the unit norm w.r.t. the specified axis.
    Actually, the module analog of `torch.nn.functional.normalize`
    """

    def __init__(self, dim=1, p=2, eps=1e-12):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.p = p

    def forward(self, tensor):
        return F.normalize(tensor, p=self.p, dim=self.dim, eps=self.eps)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
