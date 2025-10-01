import torch
import torch.nn as nn
import torch.utils.cpp_extension


class MiniBatchStd(nn.Module):
    """
    minibatch standard deviation
    """

    def forward(self, x):
        std = torch.std(x).expand(x.shape[0], 1, *x.shape[2:])
        return torch.cat([x, std], dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
