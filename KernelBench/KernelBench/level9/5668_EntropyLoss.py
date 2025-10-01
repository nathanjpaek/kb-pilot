import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


class EntropyLoss(nn.Module):
    """ Module to compute entropy loss """

    def __init__(self, normalize):
        super(EntropyLoss, self).__init__()
        self.normalize = normalize

    def forward(self, x):
        eps = 1e-05
        b = F.softmax(x, dim=1) * torch.log2(F.softmax(x, dim=1) + eps)
        b = b.sum(-1)
        if self.normalize:
            b = torch.div(b, np.log2(x.shape[1]))
        b = -1.0 * b.mean()
        return b


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'normalize': 4}]
