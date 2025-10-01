import torch
import torch.nn as nn
import torch.nn.functional as F


class softmax_SR(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        sr = F.softmax(x.reshape(x.size(0), x.size(1), -1), dim=2)
        sr = sr.transpose(1, 2)
        return sr


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
