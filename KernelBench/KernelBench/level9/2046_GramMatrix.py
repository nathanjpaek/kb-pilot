import torch
import torch.nn as nn


class GramMatrix(nn.Module):
    """
    Base Gram Matrix calculation as per Gatys et al. 2015
    """

    def forward(self, input):
        b, c, h, w = input.size()
        F = input.view(b, c, h * w)
        G = torch.bmm(F, F.transpose(1, 2))
        G = G.div_(h * w)
        return G


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
