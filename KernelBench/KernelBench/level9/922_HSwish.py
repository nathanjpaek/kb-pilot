import torch
import torch.utils.data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional
import torch.optim
import torch.nn.parallel
import torch.utils.data.distributed


class HSwish(nn.Module):
    """ Applies the Hard-Swish function element-wise.

    `"Searching for MobileNetV3" <https://arxiv.org/pdf/1905.02244.pdf>`_

    Examples:
        >>> m = Mish()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    @staticmethod
    def forward(x: 'torch.Tensor') ->torch.Tensor:
        return x * F.relu6(x + 3, inplace=True) / 6.0


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
