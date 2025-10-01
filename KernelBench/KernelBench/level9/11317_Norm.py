import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.optim.lr_scheduler import *
import torch.optim.lr_scheduler
import torch.quantization
import torch.onnx
import torch.testing


class Norm(nn.Module):
    """
    A module wrapper for vector/matrix norm
    """

    def __init__(self, p='fro', dim=None, keepdim=False):
        super(Norm, self).__init__()
        self.p = p
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: 'torch.Tensor'):
        return torch.norm(x, p=self.p, dim=self.dim, keepdim=self.keepdim)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
