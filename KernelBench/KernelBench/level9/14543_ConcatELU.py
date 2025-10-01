import torch
import torch.nn as nn
import torch.nn.functional as F


def concat_elu(x):
    """Like concatenated ReLU (http://arxiv.org/abs/1603.05201), but with ELU instead."""
    return F.elu(torch.cat([x, -x], dim=1))


class ConcatELU(nn.Module):
    """Like concatenated ReLU (http://arxiv.org/abs/1603.05201), but with ELU instead."""

    def forward(self, input):
        return concat_elu(input)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
