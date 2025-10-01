import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data


class UnitNorm(nn.Module):

    def forward(self, x):
        x = nn.functional.normalize(x, dim=1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
