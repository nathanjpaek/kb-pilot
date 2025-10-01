import torch
import torch.nn as nn
import torch.nn.functional
import torch.nn.parallel
import torch.utils.data.distributed


class Sine(nn.Module):
    """ Applies the sine function element-wise.

    `"Implicit Neural Representations with Periodic Activation Functions" <https://arxiv.org/pdf/2006.09661.pdf>`_

    Examples:
        >>> m = Sine()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    @staticmethod
    def forward(x: 'torch.Tensor') ->torch.Tensor:
        out = torch.sin(30 * x)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
