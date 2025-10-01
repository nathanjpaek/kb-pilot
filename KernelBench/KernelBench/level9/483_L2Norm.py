from torch.nn import Module
import torch
import torch.nn.functional as F


class L2Norm(Module):

    def forward(self, input):
        return F.normalize(input)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
