import torch
import numpy as np
import torch.nn as nn


class RenormSoftmax(nn.Module):

    def __init__(self, dim=-1, norm=np.pi / 40):
        super().__init__()
        self.softmax = nn.Softmax(dim=dim)
        self.dim = dim
        self.norm = norm

    def forward(self, x):
        N = x.shape[self.dim]
        return self.softmax(x) * N * self.norm


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
