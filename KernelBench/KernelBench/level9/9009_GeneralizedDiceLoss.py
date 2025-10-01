import torch
import warnings
import numpy as np
from typing import Callable
from torch.nn.modules.loss import _Loss


def one_hot(labels, num_classes):
    """
    Converts label image `labels` to a one-hot vector with `num_classes` number of channels as last dimension.
    """
    labels = labels % num_classes
    y = np.eye(num_classes)
    onehot = y[labels.flatten()]
    return onehot.reshape(tuple(labels.shape) + (num_classes,)).astype(labels
        .dtype)


class GeneralizedDiceLoss(_Loss):
    """
    Compute the generalised Dice loss defined in:

        Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
        loss function for highly unbalanced segmentations. DLMIA 2017.

    Adapted from:
        https://github.com/NifTK/NiftyNet/blob/v0.6.0/niftynet/layer/loss_segmentation.py#L279
    """

    def __init__(self, include_background: 'bool'=True, to_onehot_y: 'bool'
        =False, sigmoid: 'bool'=False, softmax: 'bool'=False, w_type: 'str'
        ='square', reduction: 'str'='mean'):
        """
        Args:
            include_background: If False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
            sigmoid: If True, apply a sigmoid function to the prediction.
            softmax: If True, apply a softmax function to the prediction.
            w_type ('square'|'simple'|'uniform'): type of function to transform ground truth volume to a weight factor.
                Default: `'square'`
            reduction (`none|mean|sum`): Specifies the reduction to apply to the output:
                ``'none'``: no reduction will be applied,
                ``'mean'``: the sum of the output will be divided by the batch size in the output,
                ``'sum'``: the output will be summed over the batch dim.
                Default: ``'mean'``.
        """
        super().__init__(reduction=reduction)
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(
                f'reduction={reduction} is invalid. Valid options are: none, mean or sum.'
                )
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        if sigmoid and softmax:
            raise ValueError(
                'sigmoid=True and softmax=True are not compatible.')
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.w_func: 'Callable' = torch.ones_like
        if w_type == 'simple':
            self.w_func = torch.reciprocal
        elif w_type == 'square':
            self.w_func = lambda x: torch.reciprocal(x * x)

    def forward(self, input: 'torch.Tensor', target: 'torch.Tensor', smooth:
        'float'=1e-05):
        """
        Args:
            input (tensor): the shape should be BNH[WD].
            target (tensor): the shape should be BNH[WD].
            smooth: a small constant to avoid nan.
        """
        if self.sigmoid:
            input = torch.sigmoid(input)
        n_pred_ch = input.shape[1]
        if n_pred_ch == 1:
            if self.softmax:
                warnings.warn(
                    'single channel prediction, `softmax=True` ignored.')
            if self.to_onehot_y:
                warnings.warn(
                    'single channel prediction, `to_onehot_y=True` ignored.')
            if not self.include_background:
                warnings.warn(
                    'single channel prediction, `include_background=False` ignored.'
                    )
        else:
            if self.softmax:
                input = torch.softmax(input, 1)
            if self.to_onehot_y:
                target = one_hot(target, n_pred_ch)
            if not self.include_background:
                target = target[:, 1:]
                input = input[:, 1:]
        assert target.shape == input.shape, f'ground truth has differing shape ({target.shape}) from input ({input.shape})'
        reduce_axis = list(range(2, len(input.shape)))
        intersection = torch.sum(target * input, reduce_axis)
        ground_o = torch.sum(target, reduce_axis)
        pred_o = torch.sum(input, reduce_axis)
        denominator = ground_o + pred_o
        w = self.w_func(ground_o.float())
        for b in w:
            infs = torch.isinf(b)
            b[infs] = 0.0
            b[infs] = torch.max(b)
        f = 1.0 - (2.0 * (intersection * w).sum(1) + smooth) / ((
            denominator * w).sum(1) + smooth)
        if self.reduction == 'mean':
            f = torch.mean(f)
        elif self.reduction == 'sum':
            f = torch.sum(f)
        elif self.reduction == 'none':
            pass
        else:
            raise ValueError(f'reduction={self.reduction} is invalid.')
        return f


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
