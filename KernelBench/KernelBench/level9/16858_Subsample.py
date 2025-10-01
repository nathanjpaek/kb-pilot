import torch
import torch.utils.data
import torch.nn as nn


class Subsample(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, feats, lengths):
        out = feats[:, ::2]
        lengths = lengths // 2
        return out, lengths


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
