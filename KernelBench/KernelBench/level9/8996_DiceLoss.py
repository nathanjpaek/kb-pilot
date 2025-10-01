import torch
import warnings
import numpy as np
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


class DiceLoss(_Loss):
    """
    Compute average Dice loss between two tensors. It can support both multi-classes and multi-labels tasks.
    Input logits `input` (BNHW[D] where N is number of classes) is compared with ground truth `target` (BNHW[D]).
    Axis N of `input` is expected to have logit predictions for each class rather than being image channels,
    while the same axis of `target` can be 1 or N (one-hot format). The `smooth` parameter is a value added to the
    intersection and union components of the inter-over-union calculation to smooth results and prevent divide by 0,
    this value should be small. The `include_background` class attribute can be set to False for an instance of
    DiceLoss to exclude the first category (channel index 0) which is by convention assumed to be background.
    If the non-background segmentations are small compared to the total image size they can get overwhelmed by
    the signal from the background so excluding it in such cases helps convergence.

    Milletari, F. et. al. (2016) V-Net: Fully Convolutional Neural Networks forVolumetric Medical Image Segmentation, 3DV, 2016.

    """

    def __init__(self, include_background: 'bool'=True, to_onehot_y: 'bool'
        =False, sigmoid: 'bool'=False, softmax: 'bool'=False, squared_pred:
        'bool'=False, jaccard: 'bool'=False, reduction: 'str'='mean'):
        """
        Args:
            include_background: If False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
            sigmoid: If True, apply a sigmoid function to the prediction.
            softmax: If True, apply a softmax function to the prediction.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction (`none|mean|sum`): Specifies the reduction to apply to the output:
                ``'none'``: no reduction will be applied,
                ``'mean'``: the sum of the output will be divided by the number of elements in the output,
                ``'sum'``: the output will be summed.
                Default: ``'mean'``.
        """
        super().__init__(reduction=reduction)
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(
                f'reduction={reduction} is invalid. Valid options are: none, mean or sum.'
                )
        if sigmoid and softmax:
            raise ValueError(
                'do_sigmoid=True and do_softmax=True are not compatible.')
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.squared_pred = squared_pred
        self.jaccard = jaccard

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
                target = one_hot(target, num_classes=n_pred_ch)
            if not self.include_background:
                target = target[:, 1:]
                input = input[:, 1:]
        assert target.shape == input.shape, f'ground truth has differing shape ({target.shape}) from input ({input.shape})'
        reduce_axis = list(range(2, len(input.shape)))
        intersection = torch.sum(target * input, dim=reduce_axis)
        if self.squared_pred:
            target = torch.pow(target, 2)
            input = torch.pow(input, 2)
        ground_o = torch.sum(target, dim=reduce_axis)
        pred_o = torch.sum(input, dim=reduce_axis)
        denominator = ground_o + pred_o
        if self.jaccard:
            denominator -= intersection
        f = 1.0 - (2.0 * intersection + smooth) / (denominator + smooth)
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
