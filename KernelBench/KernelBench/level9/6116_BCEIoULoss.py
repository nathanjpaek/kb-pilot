import torch
from typing import Callable
from functools import partial
from torch import nn
import torch.distributed
from torch.nn.modules.loss import *
from torch.nn.modules import *
from torch.optim import *
from torch.optim.lr_scheduler import *
import torch.backends


def get_activation_fn(activation: 'str'=None):
    """Returns the activation function from ``torch.nn`` by its name."""
    if activation is None or activation.lower() == 'none':

        def activation_fn(x):
            return x
    else:
        activation_fn = torch.nn.__dict__[activation]()
    return activation_fn


def wrap_metric_fn_with_activation(metric_fn: 'Callable', activation: 'str'
    =None):
    """Wraps model outputs for ``metric_fn` with specified ``activation``.

    Args:
        metric_fn: metric function to compute
        activation: activation name to use

    Returns:
        wrapped metric function with wrapped model outputs

    .. note::
        Works only with ``metric_fn`` like
        ``metric_fn(outputs, targets, *args, **kwargs)``.
    """
    activation_fn = get_activation_fn(activation)

    def wrapped_metric_fn(outputs: 'torch.Tensor', targets: 'torch.Tensor',
        *args, **kwargs):
        outputs = activation_fn(outputs)
        output = metric_fn(outputs, targets, *args, **kwargs)
        return output
    return wrapped_metric_fn


def iou(outputs: 'torch.Tensor', targets: 'torch.Tensor', class_dim: 'int'=
    1, threshold: 'float'=None, eps: 'float'=1e-07) ->torch.Tensor:
    """Computes the iou/jaccard score.

    Args:
        outputs: [N; K; ...] tensor that for each of the N examples
            indicates the probability of the example belonging to each of
            the K classes, according to the model.
        targets:  binary [N; K; ...] tensort that encodes which of the K
            classes are associated with the N-th input
        class_dim: indicates class dimention (K) for
            ``outputs`` and ``targets`` tensors (default = 1)
        threshold: threshold for outputs binarization
        eps: epsilon to avoid zero division

    Returns:
        IoU (Jaccard) score

    Examples:
        >>> size = 4
        >>> half_size = size // 2
        >>> shape = (1, 1, size, size)
        >>> empty = torch.zeros(shape)
        >>> full = torch.ones(shape)
        >>> left = torch.ones(shape)
        >>> left[:, :, :, half_size:] = 0
        >>> right = torch.ones(shape)
        >>> right[:, :, :, :half_size] = 0
        >>> top_left = torch.zeros(shape)
        >>> top_left[:, :, :half_size, :half_size] = 1
        >>> pred = torch.cat([empty, left, empty, full, left, top_left], dim=1)
        >>> targets = torch.cat([full, right, empty, full, left, left], dim=1)
        >>> iou(
        >>>     outputs=pred,
        >>>     targets=targets,
        >>>     class_dim=1,
        >>>     threshold=0.5,
        >>> )
        tensor([0.0000, 0.0000, 1.0000, 1.0000, 1.0000, 0.5])
    """
    if threshold is not None:
        outputs = (outputs > threshold).float()
    num_dims = len(outputs.shape)
    assert num_dims > 2, 'shape mismatch, please check the docs for more info'
    assert outputs.shape == targets.shape, 'shape mismatch, please check the docs for more info'
    dims = list(range(num_dims))
    if class_dim < 0:
        class_dim = num_dims + class_dim
    dims.pop(class_dim)
    sum_fn = partial(torch.sum, dim=dims)
    intersection = sum_fn(targets * outputs)
    union = sum_fn(targets) + sum_fn(outputs)
    iou_score = (intersection + eps * (union == 0).float()) / (union -
        intersection + eps)
    return iou_score


class IoULoss(nn.Module):
    """The intersection over union (Jaccard) loss.

    @TODO: Docs. Contribution is welcome.
    """

    def __init__(self, eps: 'float'=1e-07, threshold: 'float'=None,
        activation: 'str'='Sigmoid'):
        """
        Args:
            eps: epsilon to avoid zero division
            threshold: threshold for outputs binarization
            activation: An torch.nn activation applied to the outputs.
                Must be one of ``'none'``, ``'Sigmoid'``, ``'Softmax'``
        """
        super().__init__()
        metric_fn = wrap_metric_fn_with_activation(metric_fn=iou,
            activation=activation)
        self.loss_fn = partial(metric_fn, eps=eps, threshold=threshold)

    def forward(self, outputs, targets):
        """@TODO: Docs. Contribution is welcome."""
        per_class_iou = self.loss_fn(outputs, targets)
        iou = torch.mean(per_class_iou)
        return 1 - iou


class BCEIoULoss(nn.Module):
    """The Intersection over union (Jaccard) with BCE loss.

    @TODO: Docs. Contribution is welcome.
    """

    def __init__(self, eps: 'float'=1e-07, threshold: 'float'=None,
        activation: 'str'='Sigmoid', reduction: 'str'='mean'):
        """
        Args:
            eps: epsilon to avoid zero division
            threshold: threshold for outputs binarization
            activation: An torch.nn activation applied to the outputs.
                Must be one of ``'none'``, ``'Sigmoid'``, ``'Softmax'``
            reduction: Specifies the reduction to apply
                to the output of BCE
        """
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction=reduction)
        self.iou_loss = IoULoss(eps, threshold, activation)

    def forward(self, outputs, targets):
        """@TODO: Docs. Contribution is welcome."""
        iou = self.iou_loss.forward(outputs, targets)
        bce = self.bce_loss(outputs, targets)
        loss = iou + bce
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
