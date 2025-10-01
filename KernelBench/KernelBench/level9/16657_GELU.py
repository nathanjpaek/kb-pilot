import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch._utils
import torch.optim


class GELU(nn.Module):
    """Applies the Gaussian Error Linear Units function (w/ dummy inplace arg)
    """

    def __init__(self, inplace: 'bool'=False):
        super(GELU, self).__init__()

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        return F.gelu(input)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
