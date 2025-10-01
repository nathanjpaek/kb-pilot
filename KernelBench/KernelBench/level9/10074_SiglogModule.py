import torch
import torch.nn as nn


def siglog(v):
    return v.sign() * torch.log(1 + v.abs())


class SiglogModule(nn.Module):

    def forward(self, v):
        return siglog(v)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
