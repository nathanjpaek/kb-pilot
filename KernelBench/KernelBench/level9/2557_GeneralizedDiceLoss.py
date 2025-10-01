import collections
import torch
import warnings
from typing import Optional
from typing import Union
from typing import Any
from typing import Callable
from typing import Tuple
import torch.nn
from torch.nn.modules.loss import _Loss
from enum import Enum
import collections.abc


def issequenceiterable(obj: 'Any') ->bool:
    """
    Determine if the object is an iterable sequence and is not a string.
    """
    if torch.is_tensor(obj):
        return int(obj.dim()) > 0
    return isinstance(obj, collections.abc.Iterable) and not isinstance(obj,
        str)


def ensure_tuple(vals: 'Any') ->Tuple[Any, ...]:
    """
    Returns a tuple of `vals`.
    """
    if not issequenceiterable(vals):
        vals = vals,
    return tuple(vals)


def ensure_tuple_size(tup: 'Any', dim: 'int', pad_val: 'Any'=0) ->Tuple[Any,
    ...]:
    """
    Returns a copy of `tup` with `dim` values by either shortened or padded with `pad_val` as necessary.
    """
    tup = ensure_tuple(tup) + (pad_val,) * dim
    return tuple(tup[:dim])


def one_hot(labels: 'torch.Tensor', num_classes: 'int', dtype:
    'torch.dtype'=torch.float, dim: 'int'=1) ->torch.Tensor:
    """
    For a tensor `labels` of dimensions B1[spatial_dims], return a tensor of dimensions `BN[spatial_dims]`
    for `num_classes` N number of classes.

    Example:

        For every value v = labels[b,1,h,w], the value in the result at [b,v,h,w] will be 1 and all others 0.
        Note that this will include the background label, thus a binary mask should be treated as having 2 classes.
    """
    assert labels.dim() > 0, 'labels should have dim of 1 or more.'
    if labels.ndim < dim + 1:
        shape = ensure_tuple_size(labels.shape, dim + 1, 1)
        labels = labels.reshape(*shape)
    sh = list(labels.shape)
    assert sh[dim
        ] == 1, 'labels should have a channel with length equals to one.'
    sh[dim] = num_classes
    o = torch.zeros(size=sh, dtype=dtype, device=labels.device)
    labels = o.scatter_(dim=dim, index=labels.long(), value=1)
    return labels


class LossReduction(Enum):
    """
    See also:
        - :py:class:`monai.losses.dice.DiceLoss`
        - :py:class:`monai.losses.dice.GeneralizedDiceLoss`
        - :py:class:`monai.losses.focal_loss.FocalLoss`
        - :py:class:`monai.losses.tversky.TverskyLoss`
    """
    NONE = 'none'
    MEAN = 'mean'
    SUM = 'sum'


class Weight(Enum):
    """
    See also: :py:class:`monai.losses.dice.GeneralizedDiceLoss`
    """
    SQUARE = 'square'
    SIMPLE = 'simple'
    UNIFORM = 'uniform'


class GeneralizedDiceLoss(_Loss):
    """
    Compute the generalised Dice loss defined in:

        Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
        loss function for highly unbalanced segmentations. DLMIA 2017.

    Adapted from:
        https://github.com/NifTK/NiftyNet/blob/v0.6.0/niftynet/layer/loss_segmentation.py#L279
    """

    def __init__(self, include_background: 'bool'=True, to_onehot_y: 'bool'
        =False, sigmoid: 'bool'=False, softmax: 'bool'=False, other_act:
        'Optional[Callable]'=None, w_type: 'Union[Weight, str]'=Weight.
        SQUARE, reduction: 'Union[LossReduction, str]'=LossReduction.MEAN,
        smooth_nr: 'float'=1e-05, smooth_dr: 'float'=1e-05, batch: 'bool'=False
        ) ->None:
        """
        Args:
            include_background: If False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
            sigmoid: If True, apply a sigmoid function to the prediction.
            softmax: If True, apply a softmax function to the prediction.
            other_act: if don't want to use `sigmoid` or `softmax`, use other callable function to execute
                other activation layers, Defaults to ``None``. for example:
                `other_act = torch.tanh`.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            w_type: {``"square"``, ``"simple"``, ``"uniform"``}
                Type of function to transform ground truth volume to a weight factor. Defaults to ``"square"``.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, intersection over union is computed from each item in the batch.

        Raises:
            TypeError: When ``other_act`` is not an ``Optional[Callable]``.
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
                Incompatible values.

        """
        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(
                f'other_act must be None or callable but is {type(other_act).__name__}.'
                )
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError(
                'Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].'
                )
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        w_type = Weight(w_type)
        self.w_func: 'Callable' = torch.ones_like
        if w_type == Weight.SIMPLE:
            self.w_func = torch.reciprocal
        elif w_type == Weight.SQUARE:
            self.w_func = lambda x: torch.reciprocal(x * x)
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.batch = batch

    def forward(self, input: 'torch.Tensor', target: 'torch.Tensor'
        ) ->torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD].

        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        """
        if self.sigmoid:
            input = torch.sigmoid(input)
        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn(
                    'single channel prediction, `softmax=True` ignored.')
            else:
                input = torch.softmax(input, 1)
        if self.other_act is not None:
            input = self.other_act(input)
        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn(
                    'single channel prediction, `to_onehot_y=True` ignored.')
            else:
                target = one_hot(target, num_classes=n_pred_ch)
        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn(
                    'single channel prediction, `include_background=False` ignored.'
                    )
            else:
                target = target[:, 1:]
                input = input[:, 1:]
        assert target.shape == input.shape, f'ground truth has differing shape ({target.shape}) from input ({input.shape})'
        reduce_axis = list(range(2, len(input.shape)))
        if self.batch:
            reduce_axis = [0] + reduce_axis
        intersection = torch.sum(target * input, reduce_axis)
        ground_o = torch.sum(target, reduce_axis)
        pred_o = torch.sum(input, reduce_axis)
        denominator = ground_o + pred_o
        w = self.w_func(ground_o.float())
        for b in w:
            infs = torch.isinf(b)
            b[infs] = 0.0
            b[infs] = torch.max(b)
        f: 'torch.Tensor' = 1.0 - (2.0 * (intersection * w).sum(1) + self.
            smooth_nr) / ((denominator * w).sum(1) + self.smooth_dr)
        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)
        elif self.reduction == LossReduction.NONE.value:
            pass
        else:
            raise ValueError(
                f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].'
                )
        return f


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
