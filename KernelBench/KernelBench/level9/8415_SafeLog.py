import torch
import torch.nn as nn


class SafeLog(nn.Module):

    def __init__(self, eps=1e-06):
        super(SafeLog, self).__init__()
        self.eps = eps

    def forward(self, X):
        return torch.log(torch.clamp(X, min=self.eps))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
