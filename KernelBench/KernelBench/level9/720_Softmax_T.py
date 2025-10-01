import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F


class Softmax_T(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(Softmax_T, self).__init__()
        self.T = T

    def forward(self, y):
        p = F.softmax(y / self.T, dim=1)
        return p


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'T': 4}]
