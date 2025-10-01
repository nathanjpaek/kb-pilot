import torch
from torch import nn
import torch.jit
import torch.nn.functional
import torch.nn
from torch.nn.functional import gelu


class GeLU(nn.Module):

    def forward(self, x):
        return gelu(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
