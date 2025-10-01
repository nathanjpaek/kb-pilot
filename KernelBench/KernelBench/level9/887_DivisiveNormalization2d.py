from torch.nn import Module
import torch
from torch import Tensor
from typing import Union
from typing import Tuple
import torch.nn.functional as F


class DivisiveNormalization2d(Module):
    """Applies a 2D divisive normalization over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`
    and output :math:`(N, C, H, W)`.

    Args:
        b_type: Type of suppressin field, must be one of (`linf`, `l1`, `l2`).
        b_size: The size of the suppression field, must be > 0.
        sigma: Constant added to suppression field, must be > 0.

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)`

    Examples::

        >>> # suppression of size=3, sigma=1
        >>> d = DivisiveNormalization2d(b_size=3, sigma=1)
        >>> input = torch.randn(20, 16, 50, 50)
        >>> output = d(input)
    """

    def __init__(self, b_type: 'str'='linf', b_size:
        'Union[int, Tuple[int, int]]'=(5, 5), sigma: 'float'=1.0) ->None:
        super(DivisiveNormalization2d, self).__init__()
        self.sigma = sigma
        if isinstance(b_size, int):
            self.b_size = b_size, b_size
        else:
            self.b_size = b_size
        self.padding = self.b_size[0] // 2, self.b_size[1] // 2
        self.b_type = b_type

    def forward(self, input: 'Tensor') ->Tensor:
        if self.b_type == 'linf':
            suppression_field = F.max_pool2d(torch.abs(input), self.b_size,
                1, self.padding, 1)
        elif self.b_type == 'l1':
            weight = torch.ones((input.shape[1], 1, self.b_size[0], self.
                b_size[1]))
            suppression_field = F.conv2d(torch.abs(input), weight=weight,
                padding=self.padding, groups=input.shape[1])
        elif self.b_type == 'l2':
            weight = torch.ones((input.shape[1], 1, self.b_size[0], self.
                b_size[1]))
            suppression_field = torch.sqrt(F.conv2d(input ** 2, weight=
                weight, padding=self.padding, groups=input.shape[1]))
        else:
            raise NotImplementedError
        return input / (self.sigma + suppression_field)

    def __repr__(self) ->str:
        s = 'DivisiveNormalization2d('
        s += f'b_type={self.b_type}, b_size={self.b_size}, sigma={self.sigma}'
        s += ')'
        return s.format(**self.__dict__)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
