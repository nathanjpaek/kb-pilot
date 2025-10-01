import torch
import torch.nn as nn


class Norm(nn.Module):

    def __init__(self, dim_seq, input_size, eps=1e-06):
        super().__init__()
        self.size = input_size
        self.seq = dim_seq
        self.alpha = nn.Parameter(torch.ones((self.size, self.seq)))
        self.bias = nn.Parameter(torch.zeros((self.size, self.seq)))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim
            =-1, keepdim=True) + self.eps) + self.bias
        return norm


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_seq': 4, 'input_size': 4}]
