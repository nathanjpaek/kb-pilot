import torch
from torch import nn


class SafeLength(nn.Module):

    def __init__(self, dim=2, keepdim=False, eps=1e-07):
        super(SafeLength, self).__init__()
        self.dim = dim
        self.keepdim = keepdim
        self.eps = eps

    def forward(self, x):
        squared_norm = torch.sum(torch.square(x), axis=self.dim, keepdim=
            self.keepdim)
        return torch.sqrt(squared_norm + self.eps)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
