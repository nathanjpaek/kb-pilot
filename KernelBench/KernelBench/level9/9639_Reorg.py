import torch
from torch import nn
import torch.utils.data


class Reorg(nn.Module):

    def forward(self, x):
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 
            1::2], x[..., 1::2, 1::2]], 1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
