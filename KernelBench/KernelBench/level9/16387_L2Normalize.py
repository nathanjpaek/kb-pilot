import torch
import torch.nn
import torch.nn.parallel
import torch.backends.cudnn
import torch.distributed
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
import torch.optim


class L2Normalize(nn.Module):

    def __init__(self, dim):
        super(L2Normalize, self).__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
