import torch
from torch import nn


class Length(nn.Module):

    def __init__(self, dim=1, keepdim=True, p='fro'):
        super(Length, self).__init__()
        self.dim = dim
        self.keepdim = keepdim
        self.p = p

    def forward(self, inputs):
        return inputs.norm(dim=self.dim, keepdim=self.keepdim, p=self.p)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
