import torch
import torch.utils.data
import torch.nn as nn


class alpha(nn.Module):

    def __init__(self, alpha_val=0):
        super(alpha, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor([alpha_val]))
        self.alpha.requires_grad = True

    def forward(self, x):
        out = torch.mul(self.alpha, x)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
