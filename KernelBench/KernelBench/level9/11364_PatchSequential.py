import math
import torch
import warnings
from typing import Dict
from typing import Optional
from typing import Tuple
import torch.nn as nn
import torch.nn.functional as F
from typing import cast
from typing import List
from typing import Union
from torch.distributions import Bernoulli
from itertools import zip_longest
from collections import OrderedDict
from typing import Any
from typing import Iterator
from typing import NamedTuple
from torch.nn.modules.utils import _pair
from math import pi


def _adapted_sampling(shape: 'Union[Tuple, torch.Size]', dist:
    'torch.distributions.Distribution', same_on_batch=False) ->torch.Tensor:
    """The uniform sampling function that accepts 'same_on_batch'.

    If same_on_batch is True, all values generated will be exactly same given a batch_size (shape[0]).
    By default, same_on_batch is set to False.
    """
    if same_on_batch:
        return dist.sample((1, *shape[1:])).repeat(shape[0], *([1] * (len(
            shape) - 1)))
    return dist.sample(shape)


def _transform_output_shape(output:
    'Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]', shape: 'Tuple'
    ) ->Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Collapse the broadcasted batch dimensions an input tensor to be the specified shape.
    Args:
        input: torch.Tensor
        shape: List/tuple of int

    Returns:
        torch.Tensor
    """
    is_tuple = isinstance(output, tuple)
    out_tensor: 'torch.Tensor'
    trans_matrix: 'Optional[torch.Tensor]'
    if is_tuple:
        out_tensor, trans_matrix = cast(Tuple[torch.Tensor, torch.Tensor],
            output)
    else:
        out_tensor = cast(torch.Tensor, output)
        trans_matrix = None
    if trans_matrix is not None:
        if len(out_tensor.shape) > len(shape):
            assert trans_matrix.shape[0
                ] == 1, f'Dimension 0 of transformation matrix is expected to be 1, got {trans_matrix.shape[0]}'
        trans_matrix = trans_matrix.squeeze(0)
    for dim in range(len(out_tensor.shape) - len(shape)):
        assert out_tensor.shape[0
            ] == 1, f'Dimension {dim} of input is expected to be 1, got {out_tensor.shape[0]}'
        out_tensor = out_tensor.squeeze(0)
    return (out_tensor, trans_matrix) if is_tuple else out_tensor


def _transform_input(input: 'torch.Tensor') ->torch.Tensor:
    """Reshape an input tensor to be (*, C, H, W). Accept either (H, W), (C, H, W) or (*, C, H, W).
    Args:
        input: torch.Tensor

    Returns:
        torch.Tensor
    """
    if not torch.is_tensor(input):
        raise TypeError(f'Input type is not a torch.Tensor. Got {type(input)}')
    if len(input.shape) not in [2, 3, 4]:
        raise ValueError(
            f'Input size must have a shape of either (H, W), (C, H, W) or (*, C, H, W). Got {input.shape}'
            )
    if len(input.shape) == 2:
        input = input.unsqueeze(0)
    if len(input.shape) == 3:
        input = input.unsqueeze(0)
    return input


def _validate_input_dtype(input: 'torch.Tensor', accepted_dtypes: 'List'
    ) ->None:
    """Check if the dtype of the input tensor is in the range of accepted_dtypes
    Args:
        input: torch.Tensor
        accepted_dtypes: List. e.g. [torch.float32, torch.float64]
    """
    if input.dtype not in accepted_dtypes:
        raise TypeError(
            f'Expected input of {accepted_dtypes}. Got {input.dtype}')


def _extract_device_dtype(tensor_list: 'List[Optional[Any]]') ->Tuple[torch
    .device, torch.dtype]:
    """Check if all the input are in the same device (only if when they are torch.Tensor).

    If so, it would return a tuple of (device, dtype). Default: (cpu, ``get_default_dtype()``).

    Returns:
        [torch.device, torch.dtype]
    """
    device, dtype = None, None
    for tensor in tensor_list:
        if tensor is not None:
            if not isinstance(tensor, (torch.Tensor,)):
                continue
            _device = tensor.device
            _dtype = tensor.dtype
            if device is None and dtype is None:
                device = _device
                dtype = _dtype
            elif device != _device or dtype != _dtype:
                raise ValueError(
                    f'Passed values are not in the same device and dtype.Got ({device}, {dtype}) and ({_device}, {_dtype}).'
                    )
    if device is None:
        device = torch.device('cpu')
    if dtype is None:
        dtype = torch.get_default_dtype()
    return device, dtype


def _joint_range_check(ranged_factor: 'torch.Tensor', name: 'str', bounds:
    'Optional[Tuple[float, float]]'=None) ->None:
    """check if bounds[0] <= ranged_factor[0] <= ranged_factor[1] <= bounds[1]"""
    if bounds is None:
        bounds = float('-inf'), float('inf')
    if ranged_factor.dim() == 1 and len(ranged_factor) == 2:
        if not bounds[0] <= ranged_factor[0] or not bounds[1] >= ranged_factor[
            1]:
            raise ValueError(
                f'{name} out of bounds. Expected inside {bounds}, got {ranged_factor}.'
                )
        if not bounds[0] <= ranged_factor[0] <= ranged_factor[1] <= bounds[1]:
            raise ValueError(
                f'{name}[0] should be smaller than {name}[1] got {ranged_factor}'
                )
    else:
        raise TypeError(
            f'{name} should be a tensor with length 2 whose values between {bounds}. Got {ranged_factor}.'
            )


def _singular_range_check(ranged_factor: 'torch.Tensor', name: 'str',
    bounds: 'Optional[Tuple[float, float]]'=None, skip_none: 'bool'=False,
    mode: 'str'='2d') ->None:
    """check if bounds[0] <= ranged_factor[0] <= bounds[1] and bounds[0] <= ranged_factor[1] <= bounds[1]"""
    if mode == '2d':
        dim_size = 2
    elif mode == '3d':
        dim_size = 3
    else:
        raise ValueError(f"'mode' shall be either 2d or 3d. Got {mode}")
    if skip_none and ranged_factor is None:
        return
    if bounds is None:
        bounds = float('-inf'), float('inf')
    if ranged_factor.dim() == 1 and len(ranged_factor) == dim_size:
        for f in ranged_factor:
            if not bounds[0] <= f <= bounds[1]:
                raise ValueError(
                    f'{name} out of bounds. Expected inside {bounds}, got {ranged_factor}.'
                    )
    else:
        raise TypeError(
            f'{name} should be a float number or a tuple with length {dim_size} whose values between {bounds}.Got {ranged_factor}'
            )


def _range_bound(factor:
    'Union[torch.Tensor, float, Tuple[float, float], List[float]]', name:
    'str', center: 'float'=0.0, bounds: 'Tuple[float, float]'=(0, float(
    'inf')), check: 'Optional[str]'='joint', device: 'torch.device'=torch.
    device('cpu'), dtype: 'torch.dtype'=torch.get_default_dtype()
    ) ->torch.Tensor:
    """Check inputs and compute the corresponding factor bounds"""
    if not isinstance(factor, torch.Tensor):
        factor = torch.tensor(factor, device=device, dtype=dtype)
    factor_bound: 'torch.Tensor'
    if factor.dim() == 0:
        if factor < 0:
            raise ValueError(
                f'If {name} is a single number number, it must be non negative. Got {factor}'
                )
        factor_bound = factor.repeat(2) * torch.tensor([-1.0, 1.0], device=
            factor.device, dtype=factor.dtype) + center
        factor_bound = factor_bound.clamp(bounds[0], bounds[1])
    else:
        factor_bound = torch.as_tensor(factor, device=device, dtype=dtype)
    if check is not None:
        if check == 'joint':
            _joint_range_check(factor_bound, name, bounds)
        elif check == 'singular':
            _singular_range_check(factor_bound, name, bounds)
        else:
            raise NotImplementedError(f"methods '{check}' not implemented.")
    return factor_bound


def adjust_brightness(input: 'torch.Tensor', brightness_factor:
    'Union[float, torch.Tensor]') ->torch.Tensor:
    """Adjust Brightness of an image.

    .. image:: _static/img/adjust_brightness.png

    This implementation aligns OpenCV, not PIL. Hence, the output differs from TorchVision.
    The input image is expected to be in the range of [0, 1].

    Args:
        input: image to be adjusted in the shape of :math:`(*, N)`.
        brightness_factor: Brightness adjust factor per element
          in the batch. 0 does not modify the input image while any other number modify the
          brightness.

    Return:
        Adjusted image in the shape of :math:`(*, N)`.

    Example:
        >>> x = torch.ones(1, 1, 2, 2)
        >>> adjust_brightness(x, 1.)
        tensor([[[[1., 1.],
                  [1., 1.]]]])

        >>> x = torch.ones(2, 5, 3, 3)
        >>> y = torch.tensor([0.25, 0.50])
        >>> adjust_brightness(x, y).shape
        torch.Size([2, 5, 3, 3])
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f'Input type is not a torch.Tensor. Got {type(input)}')
    if not isinstance(brightness_factor, (float, torch.Tensor)):
        raise TypeError(
            f'The factor should be either a float or torch.Tensor. Got {type(brightness_factor)}'
            )
    if isinstance(brightness_factor, float):
        brightness_factor = torch.tensor([brightness_factor])
    brightness_factor = brightness_factor.to(input.device)
    for _ in input.shape[1:]:
        brightness_factor = torch.unsqueeze(brightness_factor, dim=-1)
    x_adjust: 'torch.Tensor' = input + brightness_factor
    out: 'torch.Tensor' = torch.clamp(x_adjust, 0.0, 1.0)
    return out


def adjust_contrast(input: 'torch.Tensor', contrast_factor:
    'Union[float, torch.Tensor]') ->torch.Tensor:
    """Adjust Contrast of an image.

    .. image:: _static/img/adjust_contrast.png

    This implementation aligns OpenCV, not PIL. Hence, the output differs from TorchVision.
    The input image is expected to be in the range of [0, 1].

    Args:
        input: Image to be adjusted in the shape of :math:`(*, N)`.
        contrast_factor: Contrast adjust factor per element
          in the batch. 0 generates a completely black image, 1 does not modify
          the input image while any other non-negative number modify the
          brightness by this factor.

    Return:
        Adjusted image in the shape of :math:`(*, N)`.

    Example:
        >>> x = torch.ones(1, 1, 2, 2)
        >>> adjust_contrast(x, 0.5)
        tensor([[[[0.5000, 0.5000],
                  [0.5000, 0.5000]]]])

        >>> x = torch.ones(2, 5, 3, 3)
        >>> y = torch.tensor([0.65, 0.50])
        >>> adjust_contrast(x, y).shape
        torch.Size([2, 5, 3, 3])
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f'Input type is not a torch.Tensor. Got {type(input)}')
    if not isinstance(contrast_factor, (float, torch.Tensor)):
        raise TypeError(
            f'The factor should be either a float or torch.Tensor. Got {type(contrast_factor)}'
            )
    if isinstance(contrast_factor, float):
        contrast_factor = torch.tensor([contrast_factor])
    contrast_factor = contrast_factor.to(input.device)
    if (contrast_factor < 0).any():
        raise ValueError(
            f'Contrast factor must be non-negative. Got {contrast_factor}')
    for _ in input.shape[1:]:
        contrast_factor = torch.unsqueeze(contrast_factor, dim=-1)
    x_adjust: 'torch.Tensor' = input * contrast_factor
    out: 'torch.Tensor' = torch.clamp(x_adjust, 0.0, 1.0)
    return out


def adjust_hue_raw(input: 'torch.Tensor', hue_factor:
    'Union[float, torch.Tensor]') ->torch.Tensor:
    """Adjust hue of an image. Expecting input to be in hsv format already."""
    if not isinstance(input, torch.Tensor):
        raise TypeError(f'Input type is not a torch.Tensor. Got {type(input)}')
    if not isinstance(hue_factor, (float, torch.Tensor)):
        raise TypeError(
            f'The hue_factor should be a float number or torch.Tensor in the range between [-PI, PI]. Got {type(hue_factor)}'
            )
    if isinstance(hue_factor, float):
        hue_factor = torch.as_tensor(hue_factor)
    hue_factor = hue_factor
    for _ in input.shape[1:]:
        hue_factor = torch.unsqueeze(hue_factor, dim=-1)
    h, s, v = torch.chunk(input, chunks=3, dim=-3)
    divisor: 'float' = 2 * pi
    h_out: 'torch.Tensor' = torch.fmod(h + hue_factor, divisor)
    out: 'torch.Tensor' = torch.cat([h_out, s, v], dim=-3)
    return out


def hsv_to_rgb(image: 'torch.Tensor') ->torch.Tensor:
    """Convert an image from HSV to RGB.

    The H channel values are assumed to be in the range 0..2pi. S and V are in the range 0..1.

    Args:
        image: HSV Image to be converted to HSV with shape of :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape of :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = hsv_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(
            type(image)))
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError('Input size must have a shape of (*, 3, H, W). Got {}'
            .format(image.shape))
    h: 'torch.Tensor' = image[..., 0, :, :] / (2 * math.pi)
    s: 'torch.Tensor' = image[..., 1, :, :]
    v: 'torch.Tensor' = image[..., 2, :, :]
    hi: 'torch.Tensor' = torch.floor(h * 6) % 6
    f: 'torch.Tensor' = h * 6 % 6 - hi
    one: 'torch.Tensor' = torch.tensor(1.0).to(image.device)
    p: 'torch.Tensor' = v * (one - s)
    q: 'torch.Tensor' = v * (one - f * s)
    t: 'torch.Tensor' = v * (one - (one - f) * s)
    hi = hi.long()
    indices: 'torch.Tensor' = torch.stack([hi, hi + 6, hi + 12], dim=-3)
    out = torch.stack((v, q, p, p, t, v, t, v, v, q, p, p, p, p, t, v, v, q
        ), dim=-3)
    out = torch.gather(out, -3, indices)
    return out


def rgb_to_hsv(image: 'torch.Tensor', eps: 'float'=1e-06) ->torch.Tensor:
    """Convert an image from RGB to HSV.

    .. image:: _static/img/rgb_to_hsv.png

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: RGB Image to be converted to HSV with shape of :math:`(*, 3, H, W)`.
        eps: scalar to enforce numarical stability.

    Returns:
        HSV version of the image with shape of :math:`(*, 3, H, W)`.
        The H channel values are in the range 0..2pi. S and V are in the range 0..1.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_hsv(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(
            type(image)))
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError('Input size must have a shape of (*, 3, H, W). Got {}'
            .format(image.shape))
    maxc, _ = image.max(-3)
    maxc_mask = image == maxc.unsqueeze(-3)
    _, max_indices = ((maxc_mask.cumsum(-3) == 1) & maxc_mask).max(-3)
    minc: 'torch.Tensor' = image.min(-3)[0]
    v: 'torch.Tensor' = maxc
    deltac: 'torch.Tensor' = maxc - minc
    s: 'torch.Tensor' = deltac / (v + eps)
    deltac = torch.where(deltac == 0, torch.ones_like(deltac, device=deltac
        .device, dtype=deltac.dtype), deltac)
    maxc_tmp = maxc.unsqueeze(-3) - image
    rc: 'torch.Tensor' = maxc_tmp[..., 0, :, :]
    gc: 'torch.Tensor' = maxc_tmp[..., 1, :, :]
    bc: 'torch.Tensor' = maxc_tmp[..., 2, :, :]
    h = torch.stack([bc - gc, 2.0 * deltac + rc - bc, 4.0 * deltac + gc -
        rc], dim=-3)
    h = torch.gather(h, dim=-3, index=max_indices[..., None, :, :])
    h = h.squeeze(-3)
    h = h / deltac
    h = h / 6.0 % 1.0
    h = 2 * math.pi * h
    return torch.stack([h, s, v], dim=-3)


def adjust_hue(input: 'torch.Tensor', hue_factor: 'Union[float, torch.Tensor]'
    ) ->torch.Tensor:
    """Adjust hue of an image.

    .. image:: _static/img/adjust_hue.png

    The input image is expected to be an RGB image in the range of [0, 1].

    Args:
        input: Image to be adjusted in the shape of :math:`(*, 3, H, W)`.
        hue_factor: How much to shift the hue channel. Should be in [-PI, PI]. PI
          and -PI give complete reversal of hue channel in HSV space in positive and negative
          direction respectively. 0 means no shift. Therefore, both -PI and PI will give an
          image with complementary colors while 0 gives the original image.

    Return:
        Adjusted image in the shape of :math:`(*, 3, H, W)`.

    Example:
        >>> x = torch.ones(1, 3, 2, 2)
        >>> adjust_hue(x, 3.141516).shape
        torch.Size([1, 3, 2, 2])

        >>> x = torch.ones(2, 3, 3, 3)
        >>> y = torch.ones(2) * 3.141516
        >>> adjust_hue(x, y).shape
        torch.Size([2, 3, 3, 3])
    """
    x_hsv: 'torch.Tensor' = rgb_to_hsv(input)
    x_adjusted: 'torch.Tensor' = adjust_hue_raw(x_hsv, hue_factor)
    out: 'torch.Tensor' = hsv_to_rgb(x_adjusted)
    return out


def adjust_saturation_raw(input: 'torch.Tensor', saturation_factor:
    'Union[float, torch.Tensor]') ->torch.Tensor:
    """Adjust color saturation of an image. Expecting input to be in hsv format already."""
    if not isinstance(input, torch.Tensor):
        raise TypeError(f'Input type is not a torch.Tensor. Got {type(input)}')
    if not isinstance(saturation_factor, (float, torch.Tensor)):
        raise TypeError(
            f'The saturation_factor should be a float number or torch.Tensor.Got {type(saturation_factor)}'
            )
    if isinstance(saturation_factor, float):
        saturation_factor = torch.as_tensor(saturation_factor)
    saturation_factor = saturation_factor.to(input.device)
    for _ in input.shape[1:]:
        saturation_factor = torch.unsqueeze(saturation_factor, dim=-1)
    h, s, v = torch.chunk(input, chunks=3, dim=-3)
    s_out: 'torch.Tensor' = torch.clamp(s * saturation_factor, min=0, max=1)
    out: 'torch.Tensor' = torch.cat([h, s_out, v], dim=-3)
    return out


def adjust_saturation(input: 'torch.Tensor', saturation_factor:
    'Union[float, torch.Tensor]') ->torch.Tensor:
    """Adjust color saturation of an image.

    .. image:: _static/img/adjust_saturation.png

    The input image is expected to be an RGB image in the range of [0, 1].

    Args:
        input: Image/Tensor to be adjusted in the shape of :math:`(*, 3, H, W)`.
        saturation_factor: How much to adjust the saturation. 0 will give a black
          and white image, 1 will give the original image while 2 will enhance the saturation by a factor of 2.

    Return:
        Adjusted image in the shape of :math:`(*, 3, H, W)`.

    Example:
        >>> x = torch.ones(1, 3, 3, 3)
        >>> adjust_saturation(x, 2.).shape
        torch.Size([1, 3, 3, 3])

        >>> x = torch.ones(2, 3, 3, 3)
        >>> y = torch.tensor([1., 2.])
        >>> adjust_saturation(x, y).shape
        torch.Size([2, 3, 3, 3])
    """
    x_hsv: 'torch.Tensor' = rgb_to_hsv(input)
    x_adjusted: 'torch.Tensor' = adjust_saturation_raw(x_hsv, saturation_factor
        )
    out: 'torch.Tensor' = hsv_to_rgb(x_adjusted)
    return out


def _extract_tensor_patchesnd(input: 'torch.Tensor', window_sizes:
    'Tuple[int, ...]', strides: 'Tuple[int, ...]') ->torch.Tensor:
    batch_size, num_channels = input.size()[:2]
    dims = range(2, input.dim())
    for dim, patch_size, stride in zip(dims, window_sizes, strides):
        input = input.unfold(dim, patch_size, stride)
    input = input.permute(0, *dims, 1, *[(dim + len(dims)) for dim in dims]
        ).contiguous()
    return input.view(batch_size, -1, num_channels, *window_sizes)


def extract_tensor_patches(input: 'torch.Tensor', window_size:
    'Union[int, Tuple[int, int]]', stride: 'Union[int, Tuple[int, int]]'=1,
    padding: 'Union[int, Tuple[int, int]]'=0) ->torch.Tensor:
    """Function that extract patches from tensors and stack them.

    See :class:`~kornia.contrib.ExtractTensorPatches` for details.
    """
    if not torch.is_tensor(input):
        raise TypeError('Input input type is not a torch.Tensor. Got {}'.
            format(type(input)))
    if not len(input.shape) == 4:
        raise ValueError('Invalid input shape, we expect BxCxHxW. Got: {}'.
            format(input.shape))
    if padding:
        pad_vert, pad_horz = _pair(padding)
        input = F.pad(input, [pad_horz, pad_horz, pad_vert, pad_vert])
    return _extract_tensor_patchesnd(input, _pair(window_size), _pair(stride))


class _BasicAugmentationBase(nn.Module):
    """_BasicAugmentationBase base class for customized augmentation implementations.

    Plain augmentation base class without the functionality of transformation matrix calculations.
    By default, the random computations will be happened on CPU with ``torch.get_default_dtype()``.
    To change this behaviour, please use ``set_rng_device_and_dtype``.

    Args:
        p (float): probability for applying an augmentation. This param controls the augmentation
                   probabilities element-wisely.
        p_batch (float): probability for applying an augmentation to a batch. This param controls the augmentation
                         probabilities batch-wisely.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        keepdim (bool): whether to keep the output shape the same as input (True) or broadcast it
                        to the batch form (False). Default: False.
    """

    def __init__(self, p: 'float'=0.5, p_batch: 'float'=1.0, same_on_batch:
        'bool'=False, keepdim: 'bool'=False) ->None:
        super(_BasicAugmentationBase, self).__init__()
        self.p = p
        self.p_batch = p_batch
        self.same_on_batch = same_on_batch
        self.keepdim = keepdim
        self._params: 'Dict[str, torch.Tensor]' = {}
        if p != 0.0 or p != 1.0:
            self._p_gen = Bernoulli(self.p)
        if p_batch != 0.0 or p_batch != 1.0:
            self._p_batch_gen = Bernoulli(self.p_batch)
        self.set_rng_device_and_dtype(torch.device('cpu'), torch.
            get_default_dtype())

    def __repr__(self) ->str:
        return (
            f'p={self.p}, p_batch={self.p_batch}, same_on_batch={self.same_on_batch}'
            )

    def __unpack_input__(self, input: 'torch.Tensor') ->torch.Tensor:
        return input

    def __check_batching__(self, input:
        'Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]'):
        """Check if a transformation matrix is returned,
        it has to be in the same batching mode as output."""
        raise NotImplementedError

    def transform_tensor(self, input: 'torch.Tensor') ->torch.Tensor:
        """Standardize input tensors."""
        raise NotImplementedError

    def generate_parameters(self, batch_shape: 'torch.Size') ->Dict[str,
        torch.Tensor]:
        return {}

    def apply_transform(self, input: 'torch.Tensor', params:
        'Dict[str, torch.Tensor]') ->torch.Tensor:
        raise NotImplementedError

    def set_rng_device_and_dtype(self, device: 'torch.device', dtype:
        'torch.dtype') ->None:
        """Change the random generation device and dtype.

        Note:
            The generated random numbers are not reproducible across different devices and dtypes.
        """
        self.device = device
        self.dtype = dtype

    def __batch_prob_generator__(self, batch_shape: 'torch.Size', p:
        'float', p_batch: 'float', same_on_batch: 'bool') ->torch.Tensor:
        batch_prob: 'torch.Tensor'
        if p_batch == 1:
            batch_prob = torch.tensor([True])
        elif p_batch == 0:
            batch_prob = torch.tensor([False])
        else:
            batch_prob = _adapted_sampling((1,), self._p_batch_gen,
                same_on_batch).bool()
        if batch_prob.sum().item() == 1:
            elem_prob: 'torch.Tensor'
            if p == 1:
                elem_prob = torch.tensor([True] * batch_shape[0])
            elif p == 0:
                elem_prob = torch.tensor([False] * batch_shape[0])
            else:
                elem_prob = _adapted_sampling((batch_shape[0],), self.
                    _p_gen, same_on_batch).bool()
            batch_prob = batch_prob * elem_prob
        else:
            batch_prob = batch_prob.repeat(batch_shape[0])
        return batch_prob

    def forward_parameters(self, batch_shape):
        to_apply = self.__batch_prob_generator__(batch_shape, self.p, self.
            p_batch, self.same_on_batch)
        _params = self.generate_parameters(torch.Size((int(to_apply.sum().
            item()), *batch_shape[1:])))
        if _params is None:
            _params = {}
        _params['batch_prob'] = to_apply
        return _params

    def apply_func(self, input: 'torch.Tensor', params:
        'Dict[str, torch.Tensor]') ->Union[torch.Tensor, Tuple[torch.Tensor,
        torch.Tensor]]:
        input = self.transform_tensor(input)
        return self.apply_transform(input, params)

    def forward(self, input: 'torch.Tensor', params:
        'Optional[Dict[str, torch.Tensor]]'=None) ->Union[torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor]]:
        in_tensor = self.__unpack_input__(input)
        self.__check_batching__(input)
        ori_shape = in_tensor.shape
        in_tensor = self.transform_tensor(in_tensor)
        batch_shape = in_tensor.shape
        if params is None:
            params = self.forward_parameters(batch_shape)
        self._params = params
        output = self.apply_func(input, self._params)
        return _transform_output_shape(output, ori_shape
            ) if self.keepdim else output


class _AugmentationBase(_BasicAugmentationBase):
    """_AugmentationBase base class for customized augmentation implementations.

    Advanced augmentation base class with the functionality of transformation matrix calculations.

    Args:
        p (float): probability for applying an augmentation. This param controls the augmentation probabilities
                   element-wisely for a batch.
        p_batch (float): probability for applying an augmentation to a batch. This param controls the augmentation
                         probabilities batch-wisely.
        return_transform (bool): if ``True`` return the matrix describing the geometric transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        keepdim (bool): whether to keep the output shape the same as input (True) or broadcast it
                        to the batch form (False). Default: False.
    """

    def __init__(self, return_transform: 'bool'=None, same_on_batch: 'bool'
        =False, p: 'float'=0.5, p_batch: 'float'=1.0, keepdim: 'bool'=False
        ) ->None:
        super(_AugmentationBase, self).__init__(p, p_batch=p_batch,
            same_on_batch=same_on_batch, keepdim=keepdim)
        self.p = p
        self.p_batch = p_batch
        self.return_transform = return_transform

    def __repr__(self) ->str:
        return super().__repr__(
            ) + f', return_transform={self.return_transform}'

    def identity_matrix(self, input: 'torch.Tensor') ->torch.Tensor:
        raise NotImplementedError

    def compute_transformation(self, input: 'torch.Tensor', params:
        'Dict[str, torch.Tensor]') ->torch.Tensor:
        raise NotImplementedError

    def apply_transform(self, input: 'torch.Tensor', params:
        'Dict[str, torch.Tensor]', transform: 'Optional[torch.Tensor]'=None
        ) ->torch.Tensor:
        raise NotImplementedError

    def __unpack_input__(self, input:
        'Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]') ->Tuple[
        torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(input, tuple):
            in_tensor = input[0]
            in_transformation = input[1]
            return in_tensor, in_transformation
        in_tensor = input
        return in_tensor, None

    def apply_func(self, in_tensor: 'torch.Tensor', in_transform:
        'Optional[torch.Tensor]', params: 'Dict[str, torch.Tensor]',
        return_transform: 'bool'=False) ->Union[torch.Tensor, Tuple[torch.
        Tensor, torch.Tensor]]:
        to_apply = params['batch_prob']
        if torch.sum(to_apply) == 0:
            output = in_tensor
            trans_matrix = self.identity_matrix(in_tensor)
        elif torch.sum(to_apply) == len(to_apply):
            trans_matrix = self.compute_transformation(in_tensor, params)
            output = self.apply_transform(in_tensor, params, trans_matrix)
        else:
            output = in_tensor.clone()
            trans_matrix = self.identity_matrix(in_tensor)
            trans_matrix[to_apply] = self.compute_transformation(in_tensor[
                to_apply], params)
            output[to_apply] = self.apply_transform(in_tensor[to_apply],
                params, trans_matrix[to_apply])
        self._transform_matrix = trans_matrix
        if return_transform:
            out_transformation = (trans_matrix if in_transform is None else
                trans_matrix @ in_transform)
            return output, out_transformation
        if in_transform is not None:
            return output, in_transform
        return output

    def forward(self, input:
        'Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]', params:
        'Optional[Dict[str, torch.Tensor]]'=None, return_transform:
        'Optional[bool]'=None) ->Union[torch.Tensor, Tuple[torch.Tensor,
        torch.Tensor]]:
        in_tensor, in_transform = self.__unpack_input__(input)
        self.__check_batching__(input)
        ori_shape = in_tensor.shape
        in_tensor = self.transform_tensor(in_tensor)
        batch_shape = in_tensor.shape
        if return_transform is None:
            return_transform = self.return_transform
        return_transform = cast(bool, return_transform)
        if params is None:
            params = self.forward_parameters(batch_shape)
        if 'batch_prob' not in params:
            params['batch_prob'] = torch.tensor([True] * batch_shape[0])
            warnings.warn(
                '`batch_prob` is not found in params. Will assume applying on all data.'
                )
        self._params = params
        output = self.apply_func(in_tensor, in_transform, self._params,
            return_transform)
        return _transform_output_shape(output, ori_shape
            ) if self.keepdim else output


class AugmentationBase2D(_AugmentationBase):
    """AugmentationBase2D base class for customized augmentation implementations.

    For any augmentation, the implementation of "generate_parameters" and "apply_transform" are required while the
    "compute_transformation" is only required when passing "return_transform" as True.

    Args:
        p (float): probability for applying an augmentation. This param controls the augmentation probabilities
                   element-wisely for a batch.
        p_batch (float): probability for applying an augmentation to a batch. This param controls the augmentation
                         probabilities batch-wisely.
        return_transform (bool): if ``True`` return the matrix describing the geometric transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        keepdim (bool): whether to keep the output shape the same as input (True) or broadcast it
                        to the batch form (False). Default: False.
    """

    def __check_batching__(self, input:
        'Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]'):
        if isinstance(input, tuple):
            inp, mat = input
            if len(inp.shape) == 4:
                assert len(mat.shape
                    ) == 3, 'Input tensor is in batch mode but transformation matrix is not'
                assert mat.shape[0] == inp.shape[0
                    ], f'In batch dimension, input has {inp.shape[0]}but transformation matrix has {mat.shape[0]}'
            elif len(inp.shape) == 3 or len(inp.shape) == 2:
                assert len(mat.shape
                    ) == 2, 'Input tensor is in non-batch mode but transformation matrix is not'
            else:
                raise ValueError(
                    f'Unrecognized output shape. Expected 2, 3, or 4, got {len(inp.shape)}'
                    )

    def transform_tensor(self, input: 'torch.Tensor') ->torch.Tensor:
        """Convert any incoming (H, W), (C, H, W) and (B, C, H, W) into (B, C, H, W)."""
        _validate_input_dtype(input, accepted_dtypes=[torch.float16, torch.
            float32, torch.float64])
        return _transform_input(input)

    def identity_matrix(self, input) ->torch.Tensor:
        """Return 3x3 identity matrix."""
        return kornia.eye_like(3, input)


class IntensityAugmentationBase2D(AugmentationBase2D):
    """IntensityAugmentationBase2D base class for customized intensity augmentation implementations.

    For any augmentation, the implementation of "generate_parameters" and "apply_transform" are required while the
    "compute_transformation" is only required when passing "return_transform" as True.

    Args:
        p (float): probability for applying an augmentation. This param controls the augmentation probabilities
                   element-wisely for a batch.
        p_batch (float): probability for applying an augmentation to a batch. This param controls the augmentation
                         probabilities batch-wisely.
        return_transform (bool): if ``True`` return the matrix describing the geometric transformation applied to each
                                      input tensor. If ``False`` and the input is a tuple the applied transformation
                                      wont be concatenated.
        same_on_batch (bool): apply the same transformation across the batch. Default: False.
        keepdim (bool): whether to keep the output shape the same as input (True) or broadcast it
                        to the batch form (False). Default: False.
    """

    def compute_transformation(self, input: 'torch.Tensor', params:
        'Dict[str, torch.Tensor]') ->torch.Tensor:
        return self.identity_matrix(input)


class ParamItem(NamedTuple):
    name: 'str'
    data: 'Union[dict, list]'


class ImageSequential(nn.Sequential):
    """Sequential for creating kornia image processing pipeline.

    Args:
        *args : a list of kornia augmentation and image operation modules.
        same_on_batch: apply the same transformation across the batch.
            If None, it will not overwrite the function-wise settings.
        return_transform: if ``True`` return the matrix describing the transformation
            applied to each. If None, it will not overwrite the function-wise settings.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
            to the batch form (False). If None, it will not overwrite the function-wise settings.
        random_apply: randomly select a sublist (order agnostic) of args to
            apply transformation.
            If int, a fixed number of transformations will be selected.
            If (a,), x number of transformations (a <= x <= len(args)) will be selected.
            If (a, b), x number of transformations (a <= x <= b) will be selected.
            If True, the whole list of args will be processed as a sequence in a random order.
            If False, the whole list of args will be processed as a sequence in original order.

    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: the tensor (, and the transformation matrix)
            has been sequentially modified by the args.

    Examples:
        >>> import kornia
        >>> input = torch.randn(2, 3, 5, 6)
        >>> aug_list = ImageSequential(
        ...     kornia.color.BgrToRgb(),
        ...     kornia.augmentation.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0),
        ...     kornia.filters.MedianBlur((3, 3)),
        ...     kornia.augmentation.RandomAffine(360, p=1.0),
        ...     kornia.enhance.Invert(),
        ... return_transform=True,
        ... same_on_batch=True,
        ... random_apply=10,
        ... )
        >>> out = aug_list(input)
        >>> out[0].shape, out[1].shape
        (torch.Size([2, 3, 5, 6]), torch.Size([2, 3, 3]))

        Reproduce with provided params.
        >>> out2 = aug_list(input, params=aug_list._params)
        >>> torch.equal(out[0], out2[0]), torch.equal(out[1], out2[1])
        (True, True)

    Note:
        Transformation matrix returned only considers the transformation applied in ``kornia.augmentation`` module.
        Those transformations in ``kornia.geometry`` will not be taken into account.
    """

    def __init__(self, *args: nn.Module, same_on_batch: Optional[bool]=None,
        return_transform: Optional[bool]=None, keepdim: Optional[bool]=None,
        random_apply: Union[int, bool, Tuple[int, int]]=False) ->None:
        self.same_on_batch = same_on_batch
        self.return_transform = return_transform
        self.keepdim = keepdim
        _args = OrderedDict()
        for idx, arg in enumerate(args):
            if not isinstance(arg, nn.Module):
                raise NotImplementedError(
                    f'Only nn.Module are supported at this moment. Got {arg}.')
            if isinstance(arg, _AugmentationBase):
                if same_on_batch is not None:
                    arg.same_on_batch = same_on_batch
                if return_transform is not None:
                    arg.return_transform = return_transform
                if keepdim is not None:
                    arg.keepdim = keepdim
            _args.update({f'{arg.__class__.__name__}_{idx}': arg})
        super(ImageSequential, self).__init__(_args)
        self._params: 'List[Any]' = []
        self.random_apply: 'Union[Tuple[int, int], bool]'
        if random_apply:
            if isinstance(random_apply, (bool,)) and random_apply is True:
                self.random_apply = len(args), len(args) + 1
            elif isinstance(random_apply, (int,)):
                self.random_apply = random_apply, random_apply + 1
            elif isinstance(random_apply, (tuple,)) and len(random_apply
                ) == 2 and isinstance(random_apply[0], (int,)) and isinstance(
                random_apply[1], (int,)):
                self.random_apply = random_apply[0], random_apply[1] + 1
            elif isinstance(random_apply, (tuple,)) and len(random_apply
                ) == 1 and isinstance(random_apply[0], (int,)):
                self.random_apply = random_apply[0], len(args) + 1
            else:
                raise ValueError(
                    f'Non-readable random_apply. Got {random_apply}.')
            assert isinstance(self.random_apply, (tuple,)) and len(self.
                random_apply) == 2 and isinstance(self.random_apply[0], (int,)
                ) and isinstance(self.random_apply[0], (int,)
                ), f'Expect a tuple of (int, int). Got {self.random_apply}.'
        else:
            self.random_apply = False

    def _get_child_sequence(self) ->Iterator[Tuple[str, nn.Module]]:
        if self.random_apply:
            num_samples = int(torch.randint(*self.random_apply, (1,)).item())
            indices = torch.multinomial(torch.ones((len(self),)),
                num_samples, replacement=True if num_samples > len(self) else
                False)
            return self._get_children_by_indices(indices)
        return self.named_children()

    def _get_children_by_indices(self, indices: 'torch.Tensor') ->Iterator[
        Tuple[str, nn.Module]]:
        modules = list(self.named_children())
        for idx in indices:
            yield modules[idx]

    def _get_children_by_module_names(self, names: 'List[str]') ->Iterator[
        Tuple[str, nn.Module]]:
        modules = list(self.named_children())
        for name in names:
            yield modules[list(dict(self.named_children()).keys()).index(name)]

    def get_forward_sequence(self, params: 'Optional[List[ParamItem]]'=None
        ) ->Iterator[Tuple[str, nn.Module]]:
        if params is None:
            named_modules = self._get_child_sequence()
        else:
            named_modules = self._get_children_by_module_names([p.name for
                p in params])
        return named_modules

    def apply_to_input(self, input:
        'Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]',
        module_name: 'str', module: 'Optional[nn.Module]'=None, param:
        'Optional[ParamItem]'=None) ->Union[torch.Tensor, Tuple[torch.
        Tensor, torch.Tensor]]:
        if module is None:
            module = self.get_submodule(module_name)
        if param is not None:
            assert module_name == param.name
            _param = param.data
        else:
            _param = None
        if isinstance(module, (_AugmentationBase, ImageSequential)
            ) and _param is None:
            input = module(input)
            self._params.append(ParamItem(module_name, module._params))
        elif isinstance(module, (_AugmentationBase, ImageSequential)
            ) and _param is not None:
            input = module(input, params=_param)
            self._params.append(ParamItem(module_name, _param))
        else:
            assert _param == {
                } or _param is None, f'Non-augmentaion operation {module_name} require empty parameters. Got {module}.'
            if isinstance(input, (tuple, list)):
                input = module(input[0]), input[1]
            else:
                input = module(input)
            self._params.append(ParamItem(module_name, {}))
        return input

    def forward(self, input:
        'Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]', params:
        'Optional[List[ParamItem]]'=None) ->Union[torch.Tensor, Tuple[torch
        .Tensor, torch.Tensor]]:
        self._params = []
        named_modules = self.get_forward_sequence(params)
        params = [] if params is None else params
        for (name, module), param in zip_longest(named_modules, params):
            input = self.apply_to_input(input, name, module, param=param)
        return input


class ColorJitter(IntensityAugmentationBase2D):
    """Applies a random transformation to the brightness, contrast, saturation and hue of a tensor image.

    .. image:: _static/img/ColorJitter.png

    Args:
        p: probability of applying the transformation.
        brightness: The brightness factor to apply.
        contrast: The contrast factor to apply.
        saturation: The saturation factor to apply.
        hue: The hue factor to apply.
        return_transform: if ``True`` return the matrix describing the transformation applied to each
                          input tensor. If ``False`` and the input is a tuple the applied transformation
                          wont be concatenated.
        same_on_batch: apply the same transformation across the batch.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                 to the batch form (False).

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`, Optional: :math:`(B, 3, 3)`
        - Output: :math:`(B, C, H, W)`

    Note:
        Input tensor must be float and normalized into [0, 1] for the best differentiability support.
        Additionally, this function accepts another transformation tensor (:math:`(B, 3, 3)`), then the
        applied transformation will be merged int to the input transformation tensor and returned.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> inputs = torch.ones(1, 3, 3, 3)
        >>> aug = ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.)
        >>> aug(inputs)
        tensor([[[[0.9993, 0.9993, 0.9993],
                  [0.9993, 0.9993, 0.9993],
                  [0.9993, 0.9993, 0.9993]],
        <BLANKLINE>
                 [[0.9993, 0.9993, 0.9993],
                  [0.9993, 0.9993, 0.9993],
                  [0.9993, 0.9993, 0.9993]],
        <BLANKLINE>
                 [[0.9993, 0.9993, 0.9993],
                  [0.9993, 0.9993, 0.9993],
                  [0.9993, 0.9993, 0.9993]]]])
    """

    def __init__(self, brightness:
        'Union[torch.Tensor, float, Tuple[float, float], List[float]]'=0.0,
        contrast:
        'Union[torch.Tensor, float, Tuple[float, float], List[float]]'=0.0,
        saturation:
        'Union[torch.Tensor, float, Tuple[float, float], List[float]]'=0.0,
        hue: 'Union[torch.Tensor, float, Tuple[float, float], List[float]]'
        =0.0, return_transform: 'bool'=False, same_on_batch: 'bool'=False,
        p: 'float'=1.0, keepdim: 'bool'=False) ->None:
        super(ColorJitter, self).__init__(p=p, return_transform=
            return_transform, same_on_batch=same_on_batch, keepdim=keepdim)
        self._device, self._dtype = _extract_device_dtype([brightness,
            contrast, hue, saturation])
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __repr__(self) ->str:
        repr = (
            f'brightness={self.brightness}, contrast={self.contrast}, saturation={self.saturation}, hue={self.hue}'
            )
        return self.__class__.__name__ + f'({repr}, {super().__repr__()})'

    def generate_parameters(self, batch_shape: 'torch.Size') ->Dict[str,
        torch.Tensor]:
        brightness: 'torch.Tensor' = _range_bound(self.brightness,
            'brightness', center=1.0, bounds=(0, 2), device=self._device,
            dtype=self._dtype)
        contrast: 'torch.Tensor' = _range_bound(self.contrast, 'contrast',
            center=1.0, device=self._device, dtype=self._dtype)
        saturation: 'torch.Tensor' = _range_bound(self.saturation,
            'saturation', center=1.0, device=self._device, dtype=self._dtype)
        hue: 'torch.Tensor' = _range_bound(self.hue, 'hue', bounds=(-0.5, 
            0.5), device=self._device, dtype=self._dtype)
        return rg.random_color_jitter_generator(batch_shape[0], brightness,
            contrast, saturation, hue, self.same_on_batch, self.device,
            self.dtype)

    def apply_transform(self, input: 'torch.Tensor', params:
        'Dict[str, torch.Tensor]', transform: 'Optional[torch.Tensor]'=None
        ) ->torch.Tensor:
        transforms = [lambda img: adjust_brightness(img, params[
            'brightness_factor'] - 1), lambda img: adjust_contrast(img,
            params['contrast_factor']), lambda img: adjust_saturation(img,
            params['saturation_factor']), lambda img: adjust_hue(img, 
            params['hue_factor'] * 2 * pi)]
        jittered = input
        for idx in params['order'].tolist():
            t = transforms[idx]
            jittered = t(jittered)
        return jittered


class PatchSequential(ImageSequential):
    """Container for performing patch-level image processing.

    .. image:: https://kornia-tutorials.readthedocs.io/en/latest/_images/data_patch_sequential_5_1.png

    PatchSequential breaks input images into patches by a given grid size, which will be resembled back
    afterwards. Different image processing and augmentation methods will be performed on each patch region.

    Args:
        *args: a list of processing modules.
        grid_size: controls the grid board seperation.
        padding: same or valid padding. If same padding, it will pad to include all pixels if the input
            tensor cannot be divisible by grid_size. If valid padding, the redundent border will be removed.
        same_on_batch: apply the same transformation across the batch.
            If None, it will not overwrite the function-wise settings.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
            to the batch form (False). If None, it will not overwrite the function-wise settings.
        patchwise_apply: apply image processing args will be applied patch-wisely.
            if ``True``, the number of args must be equal to grid number.
            if ``False``, the image processing args will be applied as a sequence to all patches. Default: False.
        random_apply: randomly select a sublist (order agnostic) of args to
            apply transformation.
            If ``int`` (batchwise mode only), a fixed number of transformations will be selected.
            If ``(a,)`` (batchwise mode only), x number of transformations (a <= x <= len(args)) will be selected.
            If ``(a, b)`` (batchwise mode only), x number of transformations (a <= x <= b) will be selected.
            If ``True``, the whole list of args will be processed in a random order.
            If ``False``, the whole list of args will be processed in original order.

    Return:
        List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]: the tensor (, and the transformation matrix)
            has been sequentially modified by the args.

    Examples:
        >>> import kornia.augmentation as K
        >>> input = torch.randn(2, 3, 224, 224)
        >>> seq = PatchSequential(
        ...     ImageSequential(
        ...         K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=0.5),
        ...         K.RandomPerspective(0.2, p=0.5),
        ...         K.RandomSolarize(0.1, 0.1, p=0.5),
        ...     ),
        ...     K.RandomAffine(360, p=1.0),
        ...     ImageSequential(
        ...         K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=0.5),
        ...         K.RandomPerspective(0.2, p=0.5),
        ...         K.RandomSolarize(0.1, 0.1, p=0.5),
        ...     ),
        ...     K.RandomSolarize(0.1, 0.1, p=0.1),
        ... grid_size=(2,2),
        ... patchwise_apply=False,
        ... same_on_batch=True,
        ... random_apply=True,
        ... )
        >>> out = seq(input)
        >>> out.shape
        torch.Size([2, 3, 224, 224])
        >>> out1 = seq(input, seq._params)
        >>> torch.equal(out, out1)
        True
    """

    def __init__(self, *args: nn.Module, grid_size: Tuple[int, int]=(4, 4),
        padding: str='same', same_on_batch: Optional[bool]=None, keepdim:
        Optional[bool]=None, patchwise_apply: bool=False, random_apply:
        Union[int, bool, Tuple[int, int]]=False) ->None:
        _random_apply: 'Optional[Union[int, Tuple[int, int]]]'
        if patchwise_apply and random_apply is True:
            _random_apply = grid_size[0] * grid_size[1], grid_size[0
                ] * grid_size[1]
        elif patchwise_apply and random_apply is False:
            assert len(args) == grid_size[0] * grid_size[1
                ], f'The number of processing modules must be equal with grid size.Got {len(args)} and {grid_size[0] * grid_size[1]}.'
            _random_apply = random_apply
        elif patchwise_apply and isinstance(random_apply, (int, tuple)):
            raise ValueError(
                f'Only boolean value allowed when `patchwise_apply` is set to True. Got {random_apply}.'
                )
        else:
            _random_apply = random_apply
        super(PatchSequential, self).__init__(*args, same_on_batch=
            same_on_batch, return_transform=False, keepdim=keepdim,
            random_apply=_random_apply)
        assert padding in ['same', 'valid'
            ], f'`padding` must be either `same` or `valid`. Got {padding}.'
        self.grid_size = grid_size
        self.padding = padding
        self.patchwise_apply = patchwise_apply

    def is_intensity_only(self) ->bool:
        """Check if all transformations are intensity-based.

        Note: patch processing would break the continuity of labels (e.g. bbounding boxes, masks).
        """
        for arg in self.children():
            if isinstance(arg, (ImageSequential,)):
                for _arg in arg.children():
                    if not isinstance(_arg, IntensityAugmentationBase2D):
                        return False
            elif not isinstance(_arg, IntensityAugmentationBase2D):
                return False
        return True

    def __repeat_param_across_patches__(self, param: 'torch.Tensor',
        patch_num: 'int') ->torch.Tensor:
        """Repeat parameters across patches.

        The input is shaped as (B, ...), while to output (B * patch_num, ...), which
        to guarentee that the same transformation would happen for each patch index.

        (B1, B2, ..., Bn) => (B1, ... Bn, B1, ..., Bn, ..., B1, ..., Bn)
                              | pt_size | | pt_size |  ..., | pt_size |
        """
        repeated = torch.cat([param] * patch_num, dim=0)
        return repeated

    def compute_padding(self, input: 'torch.Tensor', padding: 'str',
        grid_size: 'Optional[Tuple[int, int]]'=None) ->Tuple[int, int, int, int
        ]:
        if grid_size is None:
            grid_size = self.grid_size
        if padding == 'valid':
            ph, pw = input.size(-2) // grid_size[0], input.size(-1
                ) // grid_size[1]
            return -pw // 2, pw // 2 - pw, -ph // 2, ph // 2 - ph
        elif padding == 'same':
            ph = input.size(-2) - input.size(-2) // grid_size[0] * grid_size[0]
            pw = input.size(-1) - input.size(-1) // grid_size[1] * grid_size[1]
            return pw // 2, pw - pw // 2, ph // 2, ph - ph // 2
        else:
            raise NotImplementedError(
                f"Expect `padding` as either 'valid' or 'same'. Got {padding}."
                )

    def extract_patches(self, input: 'torch.Tensor', grid_size:
        'Optional[Tuple[int, int]]'=None, pad:
        'Optional[Tuple[int, int, int, int]]'=None) ->torch.Tensor:
        """Extract patches from tensor.

        Example:
            >>> import kornia.augmentation as K
            >>> pas = PatchSequential(K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0))
            >>> pas.extract_patches(torch.arange(16).view(1, 1, 4, 4), grid_size=(2, 2))
            tensor([[[[[ 0,  1],
                       [ 4,  5]]],
            <BLANKLINE>
            <BLANKLINE>
                     [[[ 2,  3],
                       [ 6,  7]]],
            <BLANKLINE>
            <BLANKLINE>
                     [[[ 8,  9],
                       [12, 13]]],
            <BLANKLINE>
            <BLANKLINE>
                     [[[10, 11],
                       [14, 15]]]]])
            >>> pas.extract_patches(torch.arange(54).view(1, 1, 6, 9), grid_size=(2, 2), pad=(-1, -1, -2, -2))
            tensor([[[[[19, 20, 21]]],
            <BLANKLINE>
            <BLANKLINE>
                     [[[22, 23, 24]]],
            <BLANKLINE>
            <BLANKLINE>
                     [[[28, 29, 30]]],
            <BLANKLINE>
            <BLANKLINE>
                     [[[31, 32, 33]]]]])
        """
        if pad is not None:
            input = torch.nn.functional.pad(input, list(pad))
        if grid_size is None:
            grid_size = self.grid_size
        window_size = input.size(-2) // grid_size[-2], input.size(-1
            ) // grid_size[-1]
        stride = window_size
        return extract_tensor_patches(input, window_size, stride)

    def restore_from_patches(self, patches: 'torch.Tensor', grid_size:
        'Tuple[int, int]'=(4, 4), pad:
        'Optional[Tuple[int, int, int, int]]'=None) ->torch.Tensor:
        """Restore input from patches.

        Example:
            >>> import kornia.augmentation as K
            >>> pas = PatchSequential(K.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0))
            >>> out = pas.extract_patches(torch.arange(16).view(1, 1, 4, 4), grid_size=(2, 2))
            >>> pas.restore_from_patches(out, grid_size=(2, 2))
            tensor([[[[ 0,  1,  2,  3],
                      [ 4,  5,  6,  7],
                      [ 8,  9, 10, 11],
                      [12, 13, 14, 15]]]])
        """
        if grid_size is None:
            grid_size = self.grid_size
        patches_tensor = patches.view(-1, grid_size[0], grid_size[1], *
            patches.shape[-3:])
        restored_tensor = torch.cat(torch.chunk(patches_tensor, grid_size[0
            ], dim=1), -2).squeeze(1)
        restored_tensor = torch.cat(torch.chunk(restored_tensor, grid_size[
            1], dim=1), -1).squeeze(1)
        if pad is not None:
            restored_tensor = torch.nn.functional.pad(restored_tensor, [(-i
                ) for i in pad])
        return restored_tensor

    def forward_patchwise(self, input: 'torch.Tensor', params:
        'Optional[List[List[ParamItem]]]'=None) ->torch.Tensor:
        if params is None:
            params = [[]] * input.size(1)
            auglist = [self.get_forward_sequence() for _ in range(input.
                size(1))]
        else:
            auglist = [self.get_forward_sequence(p) for p in params]
            assert input.size(0) == len(auglist) == len(params)
        out = []
        self._params = []
        for inp, proc, param in zip(input, auglist, params):
            o = []
            p = []
            for inp_pat, (proc_name, proc_pat), _param in zip_longest(inp,
                proc, param):
                if isinstance(proc_pat, (_AugmentationBase, ImageSequential)):
                    o.append(proc_pat(inp_pat[None], _param.data if _param
                         is not None else None))
                    p.append(ParamItem(proc_name, proc_pat._params))
                else:
                    o.append(proc_pat(inp_pat[None]))
                    p.append(ParamItem(proc_name, {}))
            out.append(torch.cat(o, dim=0))
            self._params.append(p)
        input = torch.stack(out, dim=0)
        return input

    def forward_batchwise(self, input: 'torch.Tensor', params:
        'Optional[List[ParamItem]]'=None) ->torch.Tensor:
        if self.same_on_batch:
            batch_shape = input.size(1), *input.shape[-3:]
            patch_num = input.size(0)
        else:
            batch_shape = input.size(0) * input.size(1), *input.shape[-3:]
        if params is None:
            params = []
            for name, aug in self.get_forward_sequence():
                if isinstance(aug, _AugmentationBase):
                    aug.same_on_batch = False
                    param = aug.forward_parameters(batch_shape)
                    if self.same_on_batch:
                        for k, v in param.items():
                            if not (k == 'order' and isinstance(aug,
                                ColorJitter)):
                                param.update({k: self.
                                    __repeat_param_across_patches__(v,
                                    patch_num)})
                    aug.same_on_batch = True
                else:
                    param = None
                params.append(ParamItem(name, param))
        input = super().forward(input.view(-1, *input.shape[-3:]), params)
        return input

    def forward(self, input:
        'Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]', params:
        'Optional[Union[List[ParamItem], List[List[ParamItem]]]]'=None
        ) ->Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Input transformation will be returned if input is a tuple."""
        if isinstance(input, (tuple,)):
            pad = self.compute_padding(input[0], self.padding)
            input = self.extract_patches(input[0], self.grid_size, pad), input[
                1]
        else:
            pad = self.compute_padding(input, self.padding)
            input = self.extract_patches(input, self.grid_size, pad)
        if not self.patchwise_apply:
            params = cast(List[ParamItem], params)
            if isinstance(input, (tuple,)):
                input = self.forward_batchwise(input[0], params), input[1]
            else:
                input = self.forward_batchwise(input, params)
        else:
            params = cast(List[List[ParamItem]], params)
            if isinstance(input, (tuple,)):
                input = self.forward_patchwise(input[0], params), input[1]
            else:
                input = self.forward_patchwise(input, params)
        if isinstance(input, (tuple,)):
            input = self.restore_from_patches(input[0], self.grid_size, pad=pad
                ), input[1]
        else:
            input = self.restore_from_patches(input, self.grid_size, pad=pad)
        return input


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
