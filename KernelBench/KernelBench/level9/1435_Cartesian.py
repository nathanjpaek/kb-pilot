import torch
from torch import nn
import torch.utils.data
import torch.utils.data.distributed
import torch.optim


class Cartesian(nn.Module):

    def forward(self, x):
        r, phi = x[..., 0], x[..., 1]
        return torch.stack((r * torch.cos(phi), r * torch.sin(phi)), dim=-1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
