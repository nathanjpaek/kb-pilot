import torch
import torch.nn as nn


class BipolarSigmoid(nn.Module):

    def forward(self, x):
        return (1.0 - torch.exp(-x)) / (1.0 + torch.exp(-x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
