import torch
import torch.utils.data
import torch.nn as nn


class soft_L1(nn.Module):

    def __init__(self):
        super(soft_L1, self).__init__()

    def forward(self, input, target, eps=0.0):
        ret = torch.abs(input - target) - eps
        ret = torch.clamp(ret, min=0.0, max=100.0)
        return ret


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
