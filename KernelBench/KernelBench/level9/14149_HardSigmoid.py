import torch
import torch.nn as nn


class HardSigmoid(nn.Module):
    """Implements the Had Mish activation module from `"H-Mish" <https://github.com/digantamisra98/H-Mish>`_
    This activation is computed as follows:
    .. math::
        f(x) = \\frac{x}{2} \\cdot \\min(2, \\max(0, x + 2))
    """

    def __init__(self, inplace: 'bool'=False) ->None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return 0.5 * (x / (1 + torch.abs(x))) + 0.5


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
