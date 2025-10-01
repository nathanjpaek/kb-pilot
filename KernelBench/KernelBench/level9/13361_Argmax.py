import torch
from torch import nn
import torch.utils.data


class Argmax(nn.Module):

    def __init__(self, dim=1, unsqueeze=True):
        super().__init__()
        self.dim = dim
        self.unsqueeze = unsqueeze

    def forward(self, x):
        argmax = torch.argmax(x, self.dim)
        if self.unsqueeze:
            argmax.unsqueeze_(1)
        return argmax


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
