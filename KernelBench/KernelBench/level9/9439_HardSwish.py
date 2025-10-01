import torch
from torch import nn
import torch.nn.functional as F


def hard_swish(x: 'torch.Tensor', inplace: 'bool'=False) ->torch.Tensor:
    inner = F.relu6(x + 3.0).div_(6.0)
    return x.mul_(inner) if inplace else x.mul(inner)


class HardSwish(nn.Module):
    """
    HardSwish activiation layer.

    Applies the hardswish function, element-wise.
    Described in: https://arxiv.org/abs/1905.02244.

    Args:
        inplace(`Bool`):
            whether use inplace version.
    Returns:
        (`torch.Tensor`)
            output tensor after activation.
    """

    def __init__(self, inplace: 'bool'=False) ->None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return hard_swish(x, self.inplace)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
