from torch.nn import Module
import torch
from torch import Tensor
from typing import Optional


def KORNIA_CHECK_IS_TENSOR(x, msg: 'Optional[str]'=None):
    if not isinstance(x, Tensor):
        raise TypeError(f'Not a Tensor type. Got: {type(x)}.\n{msg}')


def adjust_log(image: 'Tensor', gain: 'float'=1, inv: 'bool'=False,
    clip_output: 'bool'=True) ->Tensor:
    """Adjust log correction on the input image tensor.

    The input image is expected to be in the range of [0, 1].

    Reference:
    [1]: http://www.ece.ucsb.edu/Faculty/Manjunath/courses/ece178W03/EnhancePart1.pdf

    Args:
        image: Image to be adjusted in the shape of :math:`(*, H, W)`.
        gain: The multiplier of logarithmic function.
        inv:  If is set to True the function will return the inverse logarithmic correction.
        clip_output: Whether to clip the output image with range of [0, 1].

    Returns:
        Adjusted tensor in the shape of :math:`(*, H, W)`.

    Example:
        >>> x = torch.zeros(1, 1, 2, 2)
        >>> adjust_log(x, inv=True)
        tensor([[[[0., 0.],
                  [0., 0.]]]])
    """
    KORNIA_CHECK_IS_TENSOR(image, 'Expected shape (*, H, W)')
    if inv:
        img_adjust = (2 ** image - 1) * gain
    else:
        img_adjust = (1 + image).log2() * gain
    if clip_output:
        img_adjust = img_adjust.clamp(min=0.0, max=1.0)
    return img_adjust


class AdjustLog(Module):
    """Adjust log correction on the input image tensor.

    The input image is expected to be in the range of [0, 1].

    Reference:
    [1]: http://www.ece.ucsb.edu/Faculty/Manjunath/courses/ece178W03/EnhancePart1.pdf

    Args:
        image: Image to be adjusted in the shape of :math:`(*, H, W)`.
        gain: The multiplier of logarithmic function.
        inv:  If is set to True the function will return the inverse logarithmic correction.
        clip_output: Whether to clip the output image with range of [0, 1].

    Example:
        >>> x = torch.zeros(1, 1, 2, 2)
        >>> AdjustLog(inv=True)(x)
        tensor([[[[0., 0.],
                  [0., 0.]]]])
    """

    def __init__(self, gain: 'float'=1, inv: 'bool'=False, clip_output:
        'bool'=True) ->None:
        super().__init__()
        self.gain: 'float' = gain
        self.inv: 'bool' = inv
        self.clip_output: 'bool' = clip_output

    def forward(self, image: 'Tensor') ->Tensor:
        return adjust_log(image, gain=self.gain, inv=self.inv, clip_output=
            self.clip_output)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
