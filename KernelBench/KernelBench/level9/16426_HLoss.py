import torch
import torch.nn.functional as F
from torch import nn


class HLoss(nn.Module):
    """
        returning the negative entropy of an input tensor
    """

    def __init__(self, is_maximization=False):
        super(HLoss, self).__init__()
        self.is_neg = is_maximization

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        if self.is_neg:
            b = 1.0 * b.sum(dim=1).mean()
        else:
            b = -1.0 * b.sum(dim=1).mean()
        return b


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
