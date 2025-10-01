import torch
from torch import Tensor
from typing import Optional
import torch.nn as nn
from typing import List


def KORNIA_CHECK_IS_TENSOR(x, msg: 'Optional[str]'=None):
    if not isinstance(x, Tensor):
        raise TypeError(f'Not a Tensor type. Got: {type(x)}.\n{msg}')


def KORNIA_CHECK_SHAPE(x, shape: 'List[str]') ->None:
    KORNIA_CHECK_IS_TENSOR(x)
    if '*' == shape[0]:
        start_idx: 'int' = 1
        x_shape_to_check = x.shape[-len(shape) - 1:]
    else:
        start_idx = 0
        x_shape_to_check = x.shape
    for i in range(start_idx, len(shape)):
        dim_: 'str' = shape[i]
        if not dim_.isnumeric():
            continue
        dim = int(dim_)
        if x_shape_to_check[i] != dim:
            raise TypeError(
                f'{x} shape should be must be [{shape}]. Got {x.shape}')


def dog_response(input: 'torch.Tensor') ->torch.Tensor:
    """Compute the Difference-of-Gaussian response.

    Args:
        input: a given the gaussian 5d tensor :math:`(B, C, D, H, W)`.

    Return:
        the response map per channel with shape :math:`(B, C, D-1, H, W)`.

    """
    KORNIA_CHECK_SHAPE(input, ['B', 'C', 'L', 'H', 'W'])
    return input[:, :, 1:] - input[:, :, :-1]


class BlobDoG(nn.Module):
    """Module that calculates Difference-of-Gaussians blobs.

    See :func:`~kornia.feature.dog_response` for details.
    """

    def __init__(self) ->None:
        super().__init__()
        return

    def __repr__(self) ->str:
        return self.__class__.__name__

    def forward(self, input: 'torch.Tensor', sigmas:
        'Optional[torch.Tensor]'=None) ->torch.Tensor:
        return dog_response(input)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
