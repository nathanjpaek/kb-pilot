import torch
import torch.nn as nn
import torch.nn.functional
import torch.nn.parallel
import torch.utils.data.distributed


class HSigmoid(nn.Module):
    """ Applies the Hard-Sigmoid function element-wise.

    `"Searching for MobileNetV3" <https://arxiv.org/pdf/1905.02244.pdf>`_

    Examples:
        >>> m = Mish()
        >>> x = torch.randn(2)
        >>> output = m(x)
    """

    @staticmethod
    def forward(x: 'torch.Tensor') ->torch.Tensor:
        out = torch.nn.functional.relu6(x + 3, inplace=True) / 6.0
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
