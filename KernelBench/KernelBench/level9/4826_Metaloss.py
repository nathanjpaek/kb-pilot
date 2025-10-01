import torch
import torch.nn as nn
import torch.utils
import torch.utils.data.distributed


class Metaloss(nn.Module):

    def __init__(self):
        super(Metaloss, self).__init__()

    def forward(self, x):
        return x.mean(0).sum()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
