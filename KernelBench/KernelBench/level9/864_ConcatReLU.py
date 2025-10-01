import torch
import torch.nn as nn
import torch.nn.functional as F


def concat_relu(x):
    """Concatenated ReLU (http://arxiv.org/abs/1603.05201)."""
    return F.relu(torch.cat([x, -x], dim=1))


class ConcatReLU(nn.Module):
    """Concatenated ReLU (http://arxiv.org/abs/1603.05201)."""

    def forward(self, input):
        return concat_relu(input)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
