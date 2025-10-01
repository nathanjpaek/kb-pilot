import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data


class Norm(nn.Module):

    def __init__(self, dims):
        super(Norm, self).__init__()
        self.dims = dims

    def forward(self, x):
        z2 = torch.norm(x, p=2)
        out = z2 - self.dims
        out = out * out
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dims': 4}]
