import torch
from torch import nn
import torch.cuda


def hard_mish(x, inplace: 'bool'=False):
    if inplace:
        return x.mul_(0.5 * (x + 2).clamp(min=0, max=2))
    else:
        return 0.5 * x * (x + 2).clamp(min=0, max=2)


class HardMish(nn.Module):
    """
    Hard Mish
    Experimental, based on notes by Mish author Diganta Misra at
      https://github.com/digantamisra98/H-Mish/blob/0da20d4bc58e696b6803f2523c58d3c8a82782d0/README.md
    Args:
        inplace(`Bool`):
            whether use inplace version.
    Returns:
        (`torch.Tensor`)
            output tensor after activation.
    """

    def __init__(self, inplace: 'bool'=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_mish(x, self.inplace)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
