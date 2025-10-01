import torch
import torch.nn as nn


class PartitionedReLU(nn.ReLU):

    def forward(self, x):
        if isinstance(x, tuple):
            x_c, x_p = x
        else:
            x_c, x_p = torch.chunk(x, 2, dim=-1)
        return super().forward(x_c), super().forward(x_p)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
