from torch.nn import Module
import torch
from torch import Tensor
from typing import Optional


def KORNIA_CHECK_IS_TENSOR(x, msg: 'Optional[str]'=None):
    if not isinstance(x, Tensor):
        raise TypeError(f'Not a Tensor type. Got: {type(x)}.\n{msg}')


def adjust_sigmoid(image: 'Tensor', cutoff: 'float'=0.5, gain: 'float'=10,
    inv: 'bool'=False) ->Tensor:
    """Adjust sigmoid correction on the input image tensor.

    The input image is expected to be in the range of [0, 1].

    Reference:
    [1]: Gustav J. Braun, "Image Lightness Rescaling Using Sigmoidal Contrast Enhancement Functions",
        http://markfairchild.org/PDFs/PAP07.pdf

    Args:
        image: Image to be adjusted in the shape of :math:`(*, H, W)`.
        cutoff: The cutoff of sigmoid function.
        gain: The multiplier of sigmoid function.
        inv: If is set to True the function will return the inverse sigmoid correction.

    Returns:
         Adjusted tensor in the shape of :math:`(*, H, W)`.

    Example:
        >>> x = torch.ones(1, 1, 2, 2)
        >>> adjust_sigmoid(x, gain=0)
        tensor([[[[0.5000, 0.5000],
                  [0.5000, 0.5000]]]])
    """
    KORNIA_CHECK_IS_TENSOR(image, 'Expected shape (*, H, W)')
    if inv:
        img_adjust = 1 - 1 / (1 + (gain * (cutoff - image)).exp())
    else:
        img_adjust = 1 / (1 + (gain * (cutoff - image)).exp())
    return img_adjust


class AdjustSigmoid(Module):
    """Adjust the contrast of an image tensor or performs sigmoid correction on the input image tensor.

    The input image is expected to be in the range of [0, 1].

    Reference:
    [1]: Gustav J. Braun, "Image Lightness Rescaling Using Sigmoidal Contrast Enhancement Functions",
        http://markfairchild.org/PDFs/PAP07.pdf

    Args:
        image: Image to be adjusted in the shape of :math:`(*, H, W)`.
        cutoff: The cutoff of sigmoid function.
        gain: The multiplier of sigmoid function.
        inv: If is set to True the function will return the negative sigmoid correction.

    Example:
        >>> x = torch.ones(1, 1, 2, 2)
        >>> AdjustSigmoid(gain=0)(x)
        tensor([[[[0.5000, 0.5000],
                  [0.5000, 0.5000]]]])
    """

    def __init__(self, cutoff: 'float'=0.5, gain: 'float'=10, inv: 'bool'=False
        ) ->None:
        super().__init__()
        self.cutoff: 'float' = cutoff
        self.gain: 'float' = gain
        self.inv: 'bool' = inv

    def forward(self, image: 'Tensor') ->Tensor:
        return adjust_sigmoid(image, cutoff=self.cutoff, gain=self.gain,
            inv=self.inv)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
