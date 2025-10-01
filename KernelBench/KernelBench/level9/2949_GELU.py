import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU(nn.Module):

    def forward(self, input):
        return F.gelu(input)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
