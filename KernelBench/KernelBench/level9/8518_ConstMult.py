import torch
import torch.nn as nn


class ConstMult(nn.Module):

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.Tensor(1))
        nn.init.constant_(self.alpha, alpha)

    def forward(self, x):
        return self.alpha * x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
