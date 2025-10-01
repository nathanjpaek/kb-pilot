import torch
import torch.nn as nn


def sigsqrt(v):
    return v / torch.sqrt(1 + v.abs())


class SigsqrtModule(nn.Module):

    def forward(self, v):
        return sigsqrt(v)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
