import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional
from typing import Any
import torch.nn.functional as F


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".
    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.
    Returns:
        Tensor: Processed loss values.
    """
    if weight is not None:
        loss = loss * weight
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    elif reduction == 'mean':
        loss = loss.sum() / avg_factor
    elif reduction != 'none':
        raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def binary_cross_entropy(pred, label, weight=None, reduction='mean',
    avg_factor=None, class_weight=None, pos_weight=None):
    """Calculate the binary CrossEntropy loss with logits.
    Args:
        pred (torch.Tensor): The prediction with shape (N, \\*).
        label (torch.Tensor): The gt label with shape (N, \\*).
        weight (torch.Tensor, optional): Element-wise weight of loss with shape
            (N, ). Defaults to None.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". If reduction is 'none' , loss
            is same shape as pred and label. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (torch.Tensor, optional): The weight for each class with
            shape (C), C is the number of classes. Default None.
        pos_weight (torch.Tensor, optional): The positive weight for each
            class with shape (C), C is the number of classes. Default None.
    Returns:
        torch.Tensor: The calculated loss
    """
    assert pred.dim() == label.dim()
    if class_weight is not None:
        N = pred.size()[0]
        class_weight = class_weight.repeat(N, 1)
    loss = F.binary_cross_entropy_with_logits(pred, label, weight=
        class_weight, pos_weight=pos_weight, reduction='none')
    if weight is not None:
        assert weight.dim() == 1
        weight = weight.float()
        if pred.dim() > 1:
            weight = weight.reshape(-1, 1)
    loss = weight_reduce_loss(loss, weight=weight, reduction=reduction,
        avg_factor=avg_factor)
    return loss


class InterWeightedBCEWithLogits(nn.Module):

    def __init__(self, reduction: 'str'='mean', loss_weight: 'float'=1.0
        ) ->None:
        super(InterWeightedBCEWithLogits, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.register_buffer('class_weight', None)

    def receive_data_dist_info(self, num_pos_neg: 'Tensor') ->None:
        """Weight for each class is sqrt(n_c / (n_dominant + n_total))"""
        num_pos = num_pos_neg[0]
        num_dominant = num_pos.max()
        class_weight = torch.sqrt(num_pos / (num_dominant + num_pos.sum()))
        class_weight /= class_weight.sum()
        self.class_weight = class_weight

    def forward(self, cls_score: 'Tensor', label: 'Tensor', weight:
        'Optional[Tensor]'=None, avg_factor: 'Optional[float]'=None,
        reduction_override: 'Optional[str]'=None, **kwargs: Any
        ) ->torch.Tensor:
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.
            reduction)
        loss_cls = self.loss_weight * binary_cross_entropy(cls_score, label,
            weight, class_weight=self.class_weight, reduction=reduction,
            avg_factor=avg_factor, **kwargs)
        return loss_cls


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
