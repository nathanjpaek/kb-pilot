import torch
from typing import Optional
from typing import Tuple
import torch.nn as nn
from typing import Union
from torch.nn.modules.utils import _pair


def combine_tensor_patches(patches: 'torch.Tensor', window_size:
    'Tuple[int, int]'=(4, 4), stride: 'Tuple[int, int]'=(4, 4), unpadding:
    'Optional[Tuple[int, int, int, int]]'=None) ->torch.Tensor:
    """Restore input from patches.

    Args:
        patches: patched tensor.
        window_size: the size of the sliding window and the output patch size.
        stride: stride of the sliding window.
        unpadding: remove the padding added to both side of the input.

    Shape:
        - Input: :math:`(B, N, C, H_{out}, W_{out})`
        - Output: :math:`(B, C, H, W)`

    Example:
        >>> out = extract_tensor_patches(torch.arange(16).view(1, 1, 4, 4), window_size=(2, 2), stride=(2, 2))
        >>> combine_tensor_patches(out, window_size=(2, 2), stride=(2, 2))
        tensor([[[[ 0,  1,  2,  3],
                  [ 4,  5,  6,  7],
                  [ 8,  9, 10, 11],
                  [12, 13, 14, 15]]]])
    """
    if stride[0] != window_size[0] or stride[1] != window_size[1]:
        raise NotImplementedError(
            f'Only stride == window_size is supported. Got {stride} and {window_size}.Please feel free to drop a PR to Kornia Github.'
            )
    if unpadding is not None:
        window_size = window_size[0] + (unpadding[0] + unpadding[1]
            ) // window_size[0], window_size[1] + (unpadding[2] + unpadding[3]
            ) // window_size[1]
    patches_tensor = patches.view(-1, window_size[0], window_size[1], *
        patches.shape[-3:])
    restored_tensor = torch.cat(torch.chunk(patches_tensor, window_size[0],
        dim=1), -2).squeeze(1)
    restored_tensor = torch.cat(torch.chunk(restored_tensor, window_size[1],
        dim=1), -1).squeeze(1)
    if unpadding is not None:
        restored_tensor = torch.nn.functional.pad(restored_tensor, [(-i) for
            i in unpadding])
    return restored_tensor


class CombineTensorPatches(nn.Module):
    """Module that combine patches from tensors.

    In the simplest case, the output value of the operator with input size
    :math:`(B, N, C, H_{out}, W_{out})` is :math:`(B, C, H, W)`.

    where
      - :math:`B` is the batch size.
      - :math:`N` denotes the total number of extracted patches stacked in
      - :math:`C` denotes the number of input channels.
      - :math:`H`, :math:`W` the input height and width of the input in pixels.
      - :math:`H_{out}`, :math:`W_{out}` denote to denote to the patch size
        defined in the function signature.
        left-right and top-bottom order.

    * :attr:`window_size` is the size of the sliding window and controls the
      shape of the output tensor and defines the shape of the output patch.
    * :attr:`stride` controls the stride to apply to the sliding window and
      regulates the overlapping between the extracted patches.
    * :attr:`padding` controls the amount of implicit zeros-paddings on both
      sizes at each dimension.

    The parameters :attr:`window_size`, :attr:`stride` and :attr:`padding` can
    be either:

        - a single ``int`` -- in which case the same value is used for the
          height and width dimension.
        - a ``tuple`` of two ints -- in which case, the first `int` is used for
          the height dimension, and the second `int` for the width dimension.

    Args:
        patches: patched tensor.
        window_size: the size of the sliding window and the output patch size.
        unpadding: remove the padding added to both side of the input.

    Shape:
        - Input: :math:`(B, N, C, H_{out}, W_{out})`
        - Output: :math:`(B, C, H, W)`

    Example:
        >>> out = extract_tensor_patches(torch.arange(16).view(1, 1, 4, 4), window_size=(2, 2), stride=(2, 2))
        >>> combine_tensor_patches(out, window_size=(2, 2), stride=(2, 2))
        tensor([[[[ 0,  1,  2,  3],
                  [ 4,  5,  6,  7],
                  [ 8,  9, 10, 11],
                  [12, 13, 14, 15]]]])
    """

    def __init__(self, window_size: 'Union[int, Tuple[int, int]]',
        unpadding: 'Union[int, Tuple[int, int]]'=0) ->None:
        super().__init__()
        self.window_size: 'Tuple[int, int]' = _pair(window_size)
        pad: 'Tuple[int, int]' = _pair(unpadding)
        self.unpadding: 'Tuple[int, int, int, int]' = (pad[0], pad[0], pad[
            1], pad[1])

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        return combine_tensor_patches(input, self.window_size, stride=self.
            window_size, unpadding=self.unpadding)


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'window_size': 4}]
